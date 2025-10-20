from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTMAEForPreTraining
from PIL import Image, ImageDraw
import numpy as np
import base64
import io
import json

app = FastAPI(title="ViT-MAE Interactive API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor
print("Loading ViT-MAE model...")
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-huge")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-huge")
model.eval()

# Constants
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES_SIDE = IMG_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = NUM_PATCHES_SIDE ** 2  # 196

class MaskRequest(BaseModel):
    masked_indices: List[int]
    image_data: str  # base64 encoded image

class CustomViTMAEPipeline:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.config = model.config
        
    def create_custom_mask(self, batch_size: int, seq_length: int, masked_indices: list, device: str = 'cpu'):
        """Create a custom mask tensor directly"""
        mask = torch.zeros([batch_size, seq_length], device=device)
        
        # Set masked positions
        for idx in masked_indices:
            if 0 <= idx < seq_length:
                mask[:, idx] = 1
        
        # Create ids_shuffle that puts kept patches first, then masked patches
        ids_shuffle = torch.zeros([batch_size, seq_length], dtype=torch.long, device=device)
        
        kept_indices = [i for i in range(seq_length) if i not in masked_indices]
        all_indices = kept_indices + masked_indices
        
        for batch_idx in range(batch_size):
            ids_shuffle[batch_idx] = torch.tensor(all_indices, device=device)
        
        # Create ids_restore (inverse of ids_shuffle)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Get indices to keep
        len_keep = len(kept_indices)
        ids_keep = ids_shuffle[:, :len_keep]
        
        return mask, ids_restore, ids_keep
    
    def custom_forward_encoder(self, pixel_values: torch.Tensor, masked_indices: list):
        """Custom forward pass through encoder with specified masking"""
        batch_size = pixel_values.shape[0]
        
        # Get patch embeddings
        embeddings = self.model.vit.embeddings.patch_embeddings(pixel_values)
        
        # Add position embeddings
        pos_embeddings = self.model.vit.embeddings.position_embeddings
        embeddings = embeddings + pos_embeddings[:, 1:, :]
        
        # Apply custom masking
        seq_length = embeddings.shape[1]
        mask, ids_restore, ids_keep = self.create_custom_mask(
            batch_size, seq_length, masked_indices, embeddings.device
        )
        
        # Keep only visible patches
        dim = embeddings.shape[2]
        embeddings_unmasked = torch.gather(
            embeddings, dim=1, 
            index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        )
        
        # Add cls token
        cls_token = self.model.vit.embeddings.cls_token + pos_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings_unmasked.shape[0], -1, -1)
        embeddings_with_cls = torch.cat((cls_tokens, embeddings_unmasked), dim=1)
        
        # Pass through encoder
        encoder_outputs = self.model.vit.encoder(embeddings_with_cls)
        sequence_output = self.model.vit.layernorm(encoder_outputs.last_hidden_state)
        
        return sequence_output, mask, ids_restore
    
    def reconstruct_with_custom_mask(self, pixel_values: torch.Tensor, masked_indices: list):
        """Full reconstruction pipeline with custom masking"""
        with torch.no_grad():
            # Custom forward through encoder
            latent, mask, ids_restore = self.custom_forward_encoder(pixel_values, masked_indices)
            
            # Forward through decoder
            decoder_outputs = self.model.decoder(latent, ids_restore)
            logits = decoder_outputs.logits
            
            # Reconstruct image from patches
            reconstructed = self.model.unpatchify(logits)
            
            # Calculate loss on masked patches only
            target = self.model.patchify(pixel_values)
            if self.config.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5
            
            loss = (logits - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
        
        return reconstructed, mask, loss.item()

# Initialize pipeline
pipeline = CustomViTMAEPipeline(model, processor)

def decode_base64_image(image_data: str) -> Image.Image:
    """Decode base64 image data"""
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL image to base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_composite_image(original_image: Image.Image, reconstructed_tensor: torch.Tensor, 
                          masked_indices: list) -> Image.Image:
    """
    Create composite image showing original with reconstructed patches overlaid only on masked areas
    """
    # Convert tensor to image
    reconstructed_tensor = reconstructed_tensor.detach().cpu()
    # Reverter normalização do processor (ViTImageProcessor)
    mean = np.array(processor.image_mean).reshape(1, 1, 3)
    std = np.array(processor.image_std).reshape(1, 1, 3)
    reconstructed_array = reconstructed_tensor.permute(1, 2, 0).numpy()
    reconstructed_array = (reconstructed_array * std + mean)
    reconstructed_array = np.clip(reconstructed_array, 0, 1)
    reconstructed_array = (reconstructed_array * 255).round().astype(np.uint8)
    reconstructed_img = Image.fromarray(reconstructed_array, mode='RGB')

    # Resize original to match model input size
    original_resized = original_image.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')

    # Create composite by copying original and overlaying reconstructed patches
    composite = original_resized.copy()

    for patch_idx in masked_indices:
        if 0 <= patch_idx < NUM_PATCHES:
            # Calculate patch position
            row = patch_idx // NUM_PATCHES_SIDE
            col = patch_idx % NUM_PATCHES_SIDE

            x1 = col * PATCH_SIZE
            y1 = row * PATCH_SIZE
            x2 = x1 + PATCH_SIZE
            y2 = y1 + PATCH_SIZE

            # Extract patch from reconstructed image and paste to composite
            patch = reconstructed_img.crop((x1, y1, x2, y2))
            composite.paste(patch, (x1, y1))

    return composite

def create_mask_visualization(image: Image.Image, masked_indices: list) -> Image.Image:
    """Create visualization showing which patches are masked"""
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    overlay = Image.new('RGBA', (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    for i in range(NUM_PATCHES_SIDE):
        for j in range(NUM_PATCHES_SIDE):
            patch_idx = i * NUM_PATCHES_SIDE + j
            x1, y1 = j * PATCH_SIZE, i * PATCH_SIZE
            x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE
            
            # Draw grid
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 100), width=1)
            
            # Highlight masked patches
            if patch_idx in masked_indices:
                draw.rectangle([x1, y1, x2, y2], fill=(255, 100, 100, 120))
                draw.text((x1 + 2, y1 + 2), str(patch_idx), fill=(255, 255, 255, 255))
    
    result = Image.alpha_composite(image_resized.convert('RGBA'), overlay)
    return result.convert('RGB')

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the main HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ViT-MAE Interactive Reconstruction</title>
        <style>
            body {
                margin: 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                min-height: 100vh;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }
            
            .input-section, .output-section {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            
            .image-upload {
                margin-bottom: 20px;
            }
            
            .canvas-container {
                position: relative;
                display: inline-block;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 10px;
                overflow: hidden;
            }
            
            #imageCanvas, #maskCanvas {
                display: block;
                cursor: crosshair;
            }
            
            #maskCanvas {
                position: absolute;
                top: 0;
                left: 0;
                pointer-events: none;
            }
            
            .controls {
                margin: 20px 0;
            }
            
            .btn-group {
                display: flex;
                gap: 10px;
                margin: 10px 0;
                flex-wrap: wrap;
            }
            
            button {
                padding: 10px 15px;
                border: none;
                border-radius: 8px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                cursor: pointer;
                font-size: 14px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }
            
            .primary-btn {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                font-size: 16px;
                padding: 12px 25px;
            }
            
            .result-images {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-top: 20px;
            }
            
            .result-image {
                text-align: center;
            }
            
            .result-image img {
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }
            
            /* --- NOVO ESTILO PARA O BOTÃO SALVAR --- */
            #saveBtn {
                margin-top: 10px;
                background: linear-gradient(135deg, #28a745 0%, #218838 100%);
            }

            .info-panel {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                font-size: 14px;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            
            .spinner {
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-top: 3px solid white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎭 ViT-MAE Interactive Reconstruction</h1>
                <p>Click on image patches to toggle masking, then see AI reconstruct the masked areas</p>
            </div>
            
            <div class="main-content">
                <div class="input-section">
                    <h3>📤 Input & Controls</h3>
                    
                    <div class="image-upload">
                        <input type="file" id="imageInput" accept="image/*" style="margin-bottom: 10px;">
                        <button onclick="loadExampleImage()">Load Example Image</button>
                    </div>
                    
                    <div class="canvas-container">
                        <canvas id="imageCanvas" width="224" height="224"></canvas>
                        <canvas id="maskCanvas" width="224" height="224"></canvas>
                    </div>
                    
                    <div class="controls">
                        <div class="info-panel">
                            <strong>Instructions:</strong> Click on patches to toggle masking (red = masked)
                        </div>
                        
                        <div class="btn-group">
                            <button onclick="clearMask()">Clear All</button>
                            <button onclick="randomMask(25)">25% Random</button>
                            <button onclick="randomMask(50)">50% Random</button>
                            <button onclick="randomMask(75)">75% Random</button>
                        </div>
                        
                        <div class="btn-group">
                            <button onclick="centerBlock()">Center Block</button>
                            <button onclick="checkerboard()">Checkerboard</button>
                            <button onclick="cross()">Cross</button>
                            <button onclick="border()">Border</button>
                        </div>
                        
                        <button class="primary-btn" onclick="reconstruct()">🚀 Reconstruct Image</button>
                        
                        <div class="info-panel">
                            Masked patches: <span id="maskCount">0</span> / 196 (<span id="maskPercent">0</span>%)
                        </div>
                    </div>
                </div>
                
                <div class="output-section">
                    <h3>🎨 Results</h3>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Reconstructing image...</p>
                    </div>
                    
                    <div class="result-images" id="results">
                        <div class="result-image">
                            <h4>Mask Visualization</h4>
                            <img id="maskViz" src="" alt="Mask visualization will appear here">
                        </div>
                        <div class="result-image">
                            <h4>Reconstructed Image</h4>
                            <img id="reconstruction" src="" alt="Reconstruction will appear here">
                            <button id="saveBtn" onclick="saveImage()" style="display: none;">💾 Salvar Imagem</button>
                        </div>
                    </div>
                    
                    <div class="info-panel" id="resultInfo">
                        Upload an image and click patches to start!
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const PATCH_SIZE = 16;
            const NUM_PATCHES_SIDE = 14;
            const NUM_PATCHES = 196;
            
            let currentImage = null;
            let maskedPatches = new Set();
            
            const imageCanvas = document.getElementById('imageCanvas');
            const maskCanvas = document.getElementById('maskCanvas');
            const imageCtx = imageCanvas.getContext('2d');
            const maskCtx = maskCanvas.getContext('2d');
            
            // Set up canvas interaction
            imageCanvas.addEventListener('click', handleCanvasClick);
            document.getElementById('imageInput').addEventListener('change', handleImageUpload);
            
            function handleCanvasClick(event) {
                const rect = imageCanvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                
                // Convert to patch coordinates
                const patchX = Math.floor(x / PATCH_SIZE);
                const patchY = Math.floor(y / PATCH_SIZE);
                const patchIndex = patchY * NUM_PATCHES_SIDE + patchX;
                
                if (patchIndex >= 0 && patchIndex < NUM_PATCHES) {
                    if (maskedPatches.has(patchIndex)) {
                        maskedPatches.delete(patchIndex);
                    } else {
                        maskedPatches.add(patchIndex);
                    }
                    updateMaskVisualization();
                    updateMaskInfo();
                }
            }
            
            function handleImageUpload(event) {
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        loadImageToCanvas(e.target.result);
                    };
                    reader.readAsDataURL(file);
                }
            }
            
            function loadImageToCanvas(imageSrc) {
                const img = new Image();
                img.onload = function() {
                    imageCtx.drawImage(img, 0, 0, 224, 224);
                    currentImage = imageCanvas.toDataURL();
                    clearMask();
                };
                img.src = imageSrc;
            }
            
            function loadExampleImage() {
                // Load a default example image
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.onload = function() {
                    imageCtx.drawImage(img, 0, 0, 224, 224);
                    currentImage = imageCanvas.toDataURL();
                    clearMask();
                };
                img.src = 'https://images.cocodataset.org/val2017/000000039769.jpg';
            }
            
            function updateMaskVisualization() {
                maskCtx.clearRect(0, 0, 224, 224);
                
                // Draw grid
                maskCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
                maskCtx.lineWidth = 1;
                for (let i = 0; i <= NUM_PATCHES_SIDE; i++) {
                    maskCtx.beginPath();
                    maskCtx.moveTo(i * PATCH_SIZE, 0);
                    maskCtx.lineTo(i * PATCH_SIZE, 224);
                    maskCtx.stroke();
                    
                    maskCtx.beginPath();
                    maskCtx.moveTo(0, i * PATCH_SIZE);
                    maskCtx.lineTo(224, i * PATCH_SIZE);
                    maskCtx.stroke();
                }
                
                // Draw masked patches
                maskCtx.fillStyle = 'rgba(255, 100, 100, 0.6)';
                maskCtx.font = '10px Arial';
                maskCtx.textAlign = 'center';
                maskCtx.fillStyle = 'rgba(255, 100, 100, 0.6)';
                
                for (const patchIndex of maskedPatches) {
                    const row = Math.floor(patchIndex / NUM_PATCHES_SIDE);
                    const col = patchIndex % NUM_PATCHES_SIDE;
                    const x = col * PATCH_SIZE;
                    const y = row * PATCH_SIZE;
                    
                    maskCtx.fillRect(x, y, PATCH_SIZE, PATCH_SIZE);
                    
                    // Add patch number
                    maskCtx.fillStyle = 'white';
                    maskCtx.fillText(patchIndex.toString(), x + PATCH_SIZE/2, y + PATCH_SIZE/2 + 3);
                    maskCtx.fillStyle = 'rgba(255, 100, 100, 0.6)';
                }
            }
            
            function updateMaskInfo() {
                document.getElementById('maskCount').textContent = maskedPatches.size;
                document.getElementById('maskPercent').textContent = 
                    Math.round((maskedPatches.size / NUM_PATCHES) * 100);
            }
            
            function clearMask() {
                maskedPatches.clear();
                updateMaskVisualization();
                updateMaskInfo();
            }
            
            function randomMask(percentage) {
                clearMask();
                const numToMask = Math.floor(NUM_PATCHES * percentage / 100);
                const indices = Array.from({length: NUM_PATCHES}, (_, i) => i);
                
                for (let i = 0; i < numToMask; i++) {
                    const randomIndex = Math.floor(Math.random() * indices.length);
                    maskedPatches.add(indices.splice(randomIndex, 1)[0]);
                }
                
                updateMaskVisualization();
                updateMaskInfo();
            }
            
            function centerBlock() {
                clearMask();
                const center = Math.floor(NUM_PATCHES_SIDE / 2);
                for (let i = center - 2; i < center + 2; i++) {
                    for (let j = center - 2; j < center + 2; j++) {
                        if (i >= 0 && i < NUM_PATCHES_SIDE && j >= 0 && j < NUM_PATCHES_SIDE) {
                            maskedPatches.add(i * NUM_PATCHES_SIDE + j);
                        }
                    }
                }
                updateMaskVisualization();
                updateMaskInfo();
            }
            
            function checkerboard() {
                clearMask();
                for (let i = 0; i < NUM_PATCHES_SIDE; i++) {
                    for (let j = 0; j < NUM_PATCHES_SIDE; j++) {
                        if ((i + j) % 2 === 0) {
                            maskedPatches.add(i * NUM_PATCHES_SIDE + j);
                        }
                    }
                }
                updateMaskVisualization();
                updateMaskInfo();
            }
            
            function cross() {
                clearMask();
                const center = Math.floor(NUM_PATCHES_SIDE / 2);
                // Horizontal line
                for (let j = 0; j < NUM_PATCHES_SIDE; j++) {
                    maskedPatches.add(center * NUM_PATCHES_SIDE + j);
                }
                // Vertical line
                for (let i = 0; i < NUM_PATCHES_SIDE; i++) {
                    maskedPatches.add(i * NUM_PATCHES_SIDE + center);
                }
                updateMaskVisualization();
                updateMaskInfo();
            }
            
            function border() {
                clearMask();
                for (let i = 0; i < NUM_PATCHES_SIDE; i++) {
                    for (let j = 0; j < NUM_PATCHES_SIDE; j++) {
                        if (i === 0 || i === NUM_PATCHES_SIDE - 1 || 
                            j === 0 || j === NUM_PATCHES_SIDE - 1) {
                            maskedPatches.add(i * NUM_PATCHES_SIDE + j);
                        }
                    }
                }
                updateMaskVisualization();
                updateMaskInfo();
            }
            
            async function reconstruct() {
                if (!currentImage) {
                    alert('Please upload an image first!');
                    return;
                }
                
                if (maskedPatches.size === 0) {
                    alert('Please select some patches to mask!');
                    return;
                }
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('saveBtn').style.display = 'none'; // Esconde o botão ao iniciar

                try {
                    const response = await fetch('/reconstruct', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            image_data: currentImage,
                            masked_indices: Array.from(maskedPatches)
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        document.getElementById('maskViz').src = result.mask_visualization;
                        document.getElementById('reconstruction').src = result.reconstructed_image;
                        document.getElementById('resultInfo').innerHTML = 
                            `✅ Reconstruction complete!<br>
                             Loss: ${result.loss.toFixed(4)}<br>
                             Masked patches: ${result.masked_count}`;
                        
                        // --- ALTERAÇÃO: Mostra o botão de salvar ---
                        document.getElementById('saveBtn').style.display = 'block';

                    } else {
                        alert('Reconstruction failed: ' + result.error);
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('results').style.display = 'grid';
                }
            }

            // --- NOVA FUNÇÃO PARA SALVAR A IMAGEM ---
            function saveImage() {
                const link = document.createElement('a');
                link.href = document.getElementById('reconstruction').src;
                link.download = 'reconstructed_image.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            
            // Load example image on startup
            loadExampleImage();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/reconstruct")
async def reconstruct_image(request: MaskRequest):
    """Reconstruct image with custom masking"""
    try:
        # Decode image
        image = decode_base64_image(request.image_data)
        
        # Prepare input
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values
        
        # Run reconstruction
        reconstructed, mask, loss = pipeline.reconstruct_with_custom_mask(
            pixel_values, request.masked_indices
        )
        
        # Create composite image (original + reconstructed masked areas)
        composite_image = create_composite_image(image, reconstructed[0], request.masked_indices)
        
        # Create mask visualization
        mask_viz = create_mask_visualization(image, request.masked_indices)
        
        return JSONResponse({
            "success": True,
            "reconstructed_image": encode_image_to_base64(composite_image),
            "mask_visualization": encode_image_to_base64(mask_viz),
            "loss": loss,
            "masked_count": len(request.masked_indices)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    print("Starting ViT-MAE Interactive API...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)