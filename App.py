import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
import io
import zipfile
import math
import numpy as np
import colorsys
import json
import pandas as pd
from datetime import datetime, timedelta
import base64
import requests

# Google API imports (will be installed if needed)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
    import gspread
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False

# --- App Config ---
st.set_page_config(
    page_title="Instagram Grid Splitter Pro+ Advanced", 
    page_icon="🎨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .download-section {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #17a2b8;
    }
    .preview-grid {
        border: 3px dashed #667eea;
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%);
    }
    .stats-box {
        background: linear-gradient(135deg, #f0f2f6 0%, #e6e9ef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    .layer-panel {
        background: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .layer-item {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .layer-controls {
        display: flex;
        gap: 0.3rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .text-layer-controls {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .logo-layer-controls {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #dc3545;
    }
    .image-layer-controls {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .google-integration {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-banner {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .warning-banner {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .cloud-status {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .grid-selector {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafafa;
    }
    .grid-piece-selected {
        border: 3px solid #007bff !important;
        background: #e3f2fd !important;
    }
    .magic-eraser-mode {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🎨 Instagram Grid Splitter Pro+ Advanced</h1>
    <h2>with Advanced Layer Management, Multi-Image Support & Magic Tools</h2>
    <p>Professional image editing with advanced layer system, magic eraser, grid-specific editing, and intelligent grid splitting</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'layers' not in st.session_state:
    st.session_state.layers = []
if 'layer_counter' not in st.session_state:
    st.session_state.layer_counter = 0
if 'google_authenticated' not in st.session_state:
    st.session_state.google_authenticated = False
if 'drive_service' not in st.session_state:
    st.session_state.drive_service = None
if 'sheets_client' not in st.session_state:
    st.session_state.sheets_client = None
if 'project_data' not in st.session_state:
    st.session_state.project_data = {}
if 'selected_grid_pieces' not in st.session_state:
    st.session_state.selected_grid_pieces = []
if 'magic_eraser_mode' not in st.session_state:
    st.session_state.magic_eraser_mode = False
if 'editing_layer_id' not in st.session_state:
    st.session_state.editing_layer_id = None
if 'image_layers' not in st.session_state:
    st.session_state.image_layers = []

# --- Google Services Integration (same as before) ---
def authenticate_google_services(credentials_json):
    """Authenticate with Google services using service account JSON."""
    try:
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=[
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/spreadsheets'
            ]
        )
        
        # Initialize Drive service
        drive_service = build('drive', 'v3', credentials=credentials)
        
        # Initialize Sheets client
        sheets_client = gspread.authorize(credentials)
        
        return drive_service, sheets_client, True
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return None, None, False

def create_drive_folder(drive_service, folder_name, parent_folder_id=None):
    """Create a folder in Google Drive."""
    try:
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_folder_id:
            folder_metadata['parents'] = [parent_folder_id]
        
        folder = drive_service.files().create(body=folder_metadata).execute()
        return folder.get('id'), True
    except Exception as e:
        st.error(f"Failed to create folder: {str(e)}")
        return None, False

def list_drive_folders(drive_service, parent_folder_id=None):
    """List folders in Google Drive."""
    try:
        query = "mimeType='application/vnd.google-apps.folder'"
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"
        
        results = drive_service.files().list(
            q=query,
            fields="files(id, name, createdTime)"
        ).execute()
        
        return results.get('files', [])
    except Exception as e:
        st.error(f"Failed to list folders: {str(e)}")
        return []

def upload_to_drive(drive_service, file_data, filename, folder_id=None, mime_type='image/jpeg'):
    """Upload file to Google Drive."""
    try:
        file_metadata = {'name': filename}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        media = MediaIoBaseUpload(
            io.BytesIO(file_data),
            mimetype=mime_type,
            resumable=True
        )
        
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()
        
        return file.get('id'), file.get('webViewLink'), True
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None, None, False

def create_project_sheet(sheets_client, sheet_name, project_data):
    """Create a Google Sheet with project data."""
    try:
        # Create new spreadsheet
        spreadsheet = sheets_client.create(sheet_name)
        worksheet = spreadsheet.sheet1
        
        # Prepare data
        headers = ['Timestamp', 'Project Name', 'Grid Layout', 'Total Pieces', 'Output Size', 'Quality', 'Layers Count', 'Drive Folder ID', 'Notes']
        data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            project_data.get('name', 'Untitled Project'),
            project_data.get('grid_layout', 'N/A'),
            project_data.get('total_pieces', 0),
            project_data.get('output_size', 1080),
            project_data.get('quality', 95),
            project_data.get('layers_count', 0),
            project_data.get('drive_folder_id', ''),
            project_data.get('notes', '')
        ]
        
        # Write to sheet
        worksheet.append_row(headers)
        worksheet.append_row(data)
        
        return spreadsheet.id, spreadsheet.url, True
    except Exception as e:
        st.error(f"Failed to create sheet: {str(e)}")
        return None, None, False

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Main Configuration")

# --- Google Services Integration Section ---
st.sidebar.markdown('<div class="google-integration">', unsafe_allow_html=True)
st.sidebar.subheader("☁️ Google Cloud Integration")

if not GOOGLE_APIS_AVAILABLE:
    st.sidebar.error("📦 Google APIs not installed. Install with: pip install google-api-python-client google-auth gspread")
else:
    # JSON Service Account Upload
    st.sidebar.write("**🔐 Service Account Authentication**")
    credentials_file = st.sidebar.file_uploader(
        "Upload Service Account JSON",
        type=['json'],
        help="Upload your Google Cloud service account JSON file"
    )
    
    if credentials_file and not st.session_state.google_authenticated:
        credentials_json = credentials_file.read().decode('utf-8')
        
        if st.sidebar.button("🚀 Authenticate with Google"):
            with st.sidebar.spinner("Authenticating..."):
                drive_service, sheets_client, success = authenticate_google_services(credentials_json)
                
                if success:
                    st.session_state.drive_service = drive_service
                    st.session_state.sheets_client = sheets_client
                    st.session_state.google_authenticated = True
                    st.sidebar.success("✅ Google Services Connected!")
                else:
                    st.sidebar.error("❌ Authentication Failed")
    
    # Google Services Status
    if st.session_state.google_authenticated:
        st.sidebar.markdown('<div class="cloud-status">', unsafe_allow_html=True)
        st.sidebar.success("🟢 Google Drive: Connected")
        st.sidebar.success("🟢 Google Sheets: Connected")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Drive Operations
        st.sidebar.write("**📁 Google Drive Operations**")
        
        # List existing folders
        if st.sidebar.button("📂 Refresh Folders"):
            with st.sidebar.spinner("Loading folders..."):
                folders = list_drive_folders(st.session_state.drive_service)
                st.session_state.drive_folders = folders
        
        # Create new folder
        new_folder_name = st.sidebar.text_input("Create New Folder", placeholder="Enter folder name")
        if st.sidebar.button("➕ Create Folder") and new_folder_name:
            with st.sidebar.spinner("Creating folder..."):
                folder_id, success = create_drive_folder(st.session_state.drive_service, new_folder_name)
                if success:
                    st.sidebar.success(f"✅ Folder '{new_folder_name}' created!")
                    st.session_state.selected_folder_id = folder_id
        
        # Select folder for uploads
        if 'drive_folders' in st.session_state:
            folder_options = ["Root Folder"] + [f"{folder['name']} ({folder['id'][:8]}...)" for folder in st.session_state.drive_folders]
            selected_folder_idx = st.sidebar.selectbox("Select Upload Folder", range(len(folder_options)), format_func=lambda x: folder_options[x])
            
            if selected_folder_idx > 0:
                st.session_state.selected_folder_id = st.session_state.drive_folders[selected_folder_idx - 1]['id']
            else:
                st.session_state.selected_folder_id = None
        
        # Google Sheets Operations
        st.sidebar.write("**📊 Google Sheets Integration**")
        
        # Project tracking
        track_projects = st.sidebar.checkbox("📈 Track Projects in Sheets", value=True)
        
        if track_projects:
            project_sheet_name = st.sidebar.text_input("Project Sheet Name", value=f"Instagram_Projects_{datetime.now().strftime('%Y%m%d')}")
            
            # Auto-create sheet option
            auto_create_sheet = st.sidebar.checkbox("🔄 Auto-create project sheets", value=True)
    
    else:
        st.sidebar.info("🔒 Upload service account JSON to enable Google integration")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# --- Upload Section ---
st.sidebar.subheader("📁 Image Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose your main image", 
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Supported formats: JPG, PNG, WebP, BMP"
)

# Multiple additional images
st.sidebar.write("**🖼️ Additional Images**")
additional_images = st.sidebar.file_uploader(
    "Upload additional images for layers",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
    help="Upload multiple images to use as layers"
)

# Logo upload
logo_file = st.sidebar.file_uploader(
    "Upload logo/watermark (optional)", 
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Upload a logo or watermark to overlay"
)

# --- Grid Layout Options ---
st.sidebar.subheader("📐 Grid Layout")
grid_option = st.sidebar.selectbox(
    "Select grid layout:",
    [
        "1 × 2 (2 images) - Story Split",
        "1 × 3 (3 images) - Classic Horizontal", 
        "2 × 2 (4 images) - Perfect Square",
        "2 × 3 (6 images) - Rectangle Grid",
        "3 × 3 (9 images) - Ultimate Grid",
        "1 × 4 (4 images) - Extended Horizontal",
        "2 × 4 (8 images) - Wide Rectangle",
        "3 × 4 (12 images) - Large Grid",
        "4 × 4 (16 images) - Mega Grid",
        "Custom Grid"
    ],
    index=4,
    help="Choose your desired grid layout"
)

# Custom grid option
if "Custom" in grid_option:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        custom_rows = st.number_input("Rows", min_value=1, max_value=8, value=3)
    with col2:
        custom_cols = st.number_input("Cols", min_value=1, max_value=8, value=3)

# --- Advanced Image Processing Options ---
st.sidebar.subheader("🎨 Image Processing")

# Auto-square options
square_option = st.sidebar.selectbox(
    "Square formatting:",
    ["Keep original ratio", "Crop to square (center)", "Crop to square (smart)", "Pad to square", "Stretch to square", "Fill and crop"],
    index=0,
    help="How to handle non-square images"
)

# Output quality settings
st.sidebar.subheader("📊 Quality & Output")
output_size = st.sidebar.slider("Output size (px)", 400, 4320, 1080, 40, help="Standard Instagram size is 1080px, 4K is 4320px")
quality_setting = st.sidebar.slider("JPEG Quality", 60, 100, 95, 5, help="Higher = better quality, larger file")

# Image enhancements
st.sidebar.subheader("✨ Image Enhancements")
enhance_brightness = st.sidebar.slider("Brightness", 0.1, 3.0, 1.0, 0.1)
enhance_contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
enhance_saturation = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, 0.1)
enhance_sharpness = st.sidebar.slider("Sharpness", 0.1, 3.0, 1.0, 0.1)

# Advanced filters
st.sidebar.subheader("🎭 Advanced Filters")
apply_blur = st.sidebar.checkbox("Gaussian Blur")
blur_radius = st.sidebar.slider("Blur Radius", 0.1, 5.0, 1.0, 0.1) if apply_blur else 0

apply_vintage = st.sidebar.checkbox("Vintage Effect")
vintage_strength = st.sidebar.slider("Vintage Strength", 0.1, 1.0, 0.5, 0.1) if apply_vintage else 0

apply_vignette = st.sidebar.checkbox("Vignette Effect")
vignette_strength = st.sidebar.slider("Vignette Strength", 0.1, 1.0, 0.3, 0.1) if apply_vignette else 0

# Color grading
st.sidebar.subheader("🌈 Color Grading")
apply_color_grade = st.sidebar.checkbox("Color Grading")
if apply_color_grade:
    temperature = st.sidebar.slider("Temperature", -100, 100, 0)
    tint = st.sidebar.slider("Tint", -100, 100, 0)
    highlights = st.sidebar.slider("Highlights", -100, 100, 0)
    shadows = st.sidebar.slider("Shadows", -100, 100, 0)

# Magic Tools
st.sidebar.subheader("🪄 Magic Tools")
st.session_state.magic_eraser_mode = st.sidebar.checkbox("🔥 Magic Eraser Mode", value=st.session_state.magic_eraser_mode)
if st.session_state.magic_eraser_mode:
    eraser_size = st.sidebar.slider("Eraser Size", 5, 100, 20)
    eraser_strength = st.sidebar.slider("Eraser Strength", 0.1, 1.0, 0.8, 0.1)

# Border and spacing options
st.sidebar.subheader("🎯 Layout Options")
add_borders = st.sidebar.checkbox("Add borders between pieces", value=False)
border_width = st.sidebar.slider("Border width (px)", 1, 50, 10) if add_borders else 0
border_color = st.sidebar.color_picker("Border color", "#FFFFFF") if add_borders else "#FFFFFF"

# Padding options
pad_color = st.sidebar.color_picker("Padding color", "#FFFFFF")

# Output format
st.sidebar.subheader("💾 Output Options")
output_format = st.sidebar.radio("Download format:", ["PNG", "JPG", "Both"], index=1)
create_collage = st.sidebar.checkbox("Create collage preview", value=True)

# Numbering options
add_numbers = st.sidebar.checkbox("Add posting order numbers", value=True)
number_style = st.sidebar.selectbox(
    "Number style:",
    ["Red bold", "White with black outline", "Subtle gray", "Colorful", "Neon glow", "Vintage badge"],
    index=0
) if add_numbers else "Red bold"

number_position = st.sidebar.selectbox(
    "Number position:",
    ["center", "top-left", "top-right", "bottom-left", "bottom-right"],
    index=0
) if add_numbers else "center"

# --- Helper Functions ---
def get_grid_dimensions(grid_option, custom_rows=3, custom_cols=3):
    """Get rows and columns from grid option."""
    grid_map = {
        "1 × 2": (1, 2), "1 × 3": (1, 3), "2 × 2": (2, 2),
        "2 × 3": (2, 3), "3 × 3": (3, 3), "1 × 4": (1, 4),
        "2 × 4": (2, 4), "3 × 4": (3, 4), "4 × 4": (4, 4)
    }
    
    for key, value in grid_map.items():
        if key in grid_option:
            return value
    
    if "Custom" in grid_option:
        return custom_rows, custom_cols
    return 3, 3

def smart_crop_to_square(img):
    """Enhanced smart crop with face detection simulation and edge analysis."""
    w, h = img.size
    if w == h:
        return img
    
    # Convert to numpy for advanced analysis
    gray = np.array(img.convert('L'))
    
    # Simulate face detection by finding high-contrast regions in upper portion
    upper_third = gray[:h//3, :]
    upper_contrast = np.std(upper_third, axis=0)
    
    # Edge detection for interesting regions
    edges = np.array(img.convert('L').filter(ImageFilter.FIND_EDGES))
    
    if w > h:
        # Landscape: find best vertical strip
        face_scores = np.convolve(upper_contrast, np.ones(h)/(h), mode='valid')
        edge_scores = np.sum(edges, axis=0)
        edge_scores = np.convolve(edge_scores, np.ones(h)/(h), mode='valid')
        
        # Combine face and edge scores
        combined_scores = face_scores * 2 + edge_scores  # Weight faces more
        best_start = np.argmax(combined_scores)
        
        return img.crop((best_start, 0, best_start + h, h))
    else:
        # Portrait: find best horizontal strip
        edge_scores = np.sum(edges, axis=1)
        edge_scores = np.convolve(edge_scores, np.ones(w)/(w), mode='valid')
        best_start = np.argmax(edge_scores)
        
        return img.crop((0, best_start, w, best_start + w))

def make_square(img, option="Keep original ratio", size=None, pad_color="#FFFFFF"):
    """Enhanced square conversion with new options."""
    if option == "Keep original ratio":
        if size:
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img
    
    w, h = img.size
    
    if option == "Crop to square (center)":
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        img = img.crop((left, top, left + min_side, top + min_side))
    
    elif option == "Crop to square (smart)":
        img = smart_crop_to_square(img)
    
    elif option == "Pad to square":
        max_side = max(w, h)
        new_img = Image.new('RGB', (max_side, max_side), pad_color)
        offset = ((max_side - w) // 2, (max_side - h) // 2)
        new_img.paste(img, offset)
        img = new_img
    
    elif option == "Stretch to square":
        min_side = min(w, h)
        img = img.resize((min_side, min_side), Image.Resampling.LANCZOS)
    
    elif option == "Fill and crop":
        # Scale to fill square and crop excess
        max_side = max(w, h)
        if w > h:
            new_h = max_side
            new_w = int(w * (max_side / h))
        else:
            new_w = max_side
            new_h = int(h * (max_side / w))
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop to square
        left = (new_w - max_side) // 2
        top = (new_h - max_side) // 2
        img = img.crop((left, top, left + max_side, top + max_side))
    
    # Resize to final output size
    if size and img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    return img

def enhance_image(img, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    """Apply comprehensive image enhancements."""
    enhanced = img.copy()
    
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(saturation)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(sharpness)
    
    return enhanced

def apply_color_grading(img, temperature=0, tint=0, highlights=0, shadows=0):
    """Apply color grading effects."""
    if all(x == 0 for x in [temperature, tint, highlights, shadows]):
        return img
    
    # Convert to numpy array
    img_array = np.array(img).astype(np.float32)
    
    # Temperature adjustment (blue-orange)
    if temperature != 0:
        temp_factor = temperature / 100.0
        if temp_factor > 0:  # Warmer
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + temp_factor * 0.3), 0, 255)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 + temp_factor * 0.1), 0, 255)  # Green
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - temp_factor * 0.2), 0, 255)  # Blue
        else:  # Cooler
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + temp_factor * 0.2), 0, 255)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 + temp_factor * 0.1), 0, 255)  # Green
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - temp_factor * 0.3), 0, 255)  # Blue
    
    # Tint adjustment (green-magenta)
    if tint != 0:
        tint_factor = tint / 100.0
        if tint_factor > 0:  # More magenta
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + tint_factor * 0.2), 0, 255)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 + tint_factor * 0.2), 0, 255)  # Blue
        else:  # More green
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 - tint_factor * 0.2), 0, 255)  # Green
    
    # Highlights and shadows adjustment
    gray = np.dot(img_array, [0.299, 0.587, 0.114])
    
    if highlights != 0:
        highlight_mask = gray > 128
        highlight_factor = 1 + (highlights / 100.0) * 0.5
        img_array[highlight_mask] = np.clip(img_array[highlight_mask] * highlight_factor, 0, 255)
    
    if shadows != 0:
        shadow_mask = gray <= 128
        shadow_factor = 1 + (shadows / 100.0) * 0.5
        img_array[shadow_mask] = np.clip(img_array[shadow_mask] * shadow_factor, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def apply_advanced_filters(img, blur_radius=0, vintage_strength=0, vignette_strength=0):
    """Apply advanced filter effects."""
    filtered = img.copy()
    
    # Gaussian blur
    if blur_radius > 0:
        filtered = filtered.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Vintage effect
    if vintage_strength > 0:
        # Create sepia effect
        grayscale = filtered.convert('L')
        sepia = Image.new('RGB', filtered.size)
        sepia_pixels = []
        
        for pixel in grayscale.getdata():
            # Sepia tone calculation
            r = int(pixel * 1.0)
            g = int(pixel * 0.8)
            b = int(pixel * 0.6)
            sepia_pixels.append((min(255, r), min(255, g), min(255, b)))
        
        sepia.putdata(sepia_pixels)
        filtered = Image.blend(filtered, sepia, vintage_strength)
    
    # Vignette effect
    if vignette_strength > 0:
        w, h = filtered.size
        vignette_mask = Image.new('L', (w, h), 255)
        draw = ImageDraw.Draw(vignette_mask)
        
        # Create radial gradient for vignette
        center_x, center_y = w // 2, h // 2
        max_radius = math.sqrt(center_x**2 + center_y**2)
        
        for x in range(w):
            for y in range(h):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                alpha = max(0, 255 - int(255 * vignette_strength * (distance / max_radius)))
                vignette_mask.putpixel((x, y), alpha)
        
        # Apply vignette
        black_img = Image.new('RGB', (w, h), (0, 0, 0))
        filtered = Image.composite(filtered, black_img, vignette_mask)
    
    return filtered

def apply_magic_eraser(img, mask_points, eraser_size=20, strength=0.8):
    """Apply magic eraser effect to remove objects."""
    # Create a mask for the eraser
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw eraser strokes
    for point in mask_points:
        x, y = point
        draw.ellipse([x-eraser_size//2, y-eraser_size//2, x+eraser_size//2, y+eraser_size//2], fill=255)
    
    # Apply content-aware fill (simplified version)
    # In a real implementation, this would use advanced inpainting algorithms
    img_array = np.array(img)
    mask_array = np.array(mask)
    
    # Simple inpainting: blur the masked areas
    blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
    blurred_array = np.array(blurred)
    
    # Blend based on mask
    mask_normalized = mask_array / 255.0 * strength
    for c in range(3):  # RGB channels
        img_array[:, :, c] = (1 - mask_normalized) * img_array[:, :, c] + mask_normalized * blurred_array[:, :, c]
    
    return Image.fromarray(img_array.astype(np.uint8))

def resize_image_layer(img, scale_factor):
    """Resize an image layer."""
    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def crop_image_layer(img, crop_box):
    """Crop an image layer."""
    return img.crop(crop_box)

def add_text_layer(img, text, font_size=50, color=(255, 255, 255), position=(50, 50), 
                  font_style="bold", rotation=0, opacity=255, outline_color=None, outline_width=0,
                  shadow=False, shadow_offset=(5, 5), shadow_color=(0, 0, 0), grid_pieces=None):
    """Add professional text overlay with advanced styling and grid-specific placement."""
    # Create overlay for text
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to load font
    try:
        if font_style == "bold":
            font = ImageFont.truetype("arial.ttf", font_size)
        elif font_style == "italic":
            font = ImageFont.truetype("ariali.ttf", font_size)
        else:
            font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Add shadow if requested
    if shadow:
        shadow_pos = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])
        draw.text(shadow_pos, text, font=font, fill=shadow_color + (opacity//2,))
    
    # Add outline if requested
    if outline_color and outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx*dx + dy*dy <= outline_width*outline_width:
                    draw.text((position[0] + dx, position[1] + dy), text, 
                             font=font, fill=outline_color + (opacity,))
    
    # Add main text
    draw.text(position, text, font=font, fill=color + (opacity,))
    
    # Apply rotation if needed
    if rotation != 0:
        overlay = overlay.rotate(rotation, expand=True)
    
    # Composite with original image
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    return Image.alpha_composite(img, overlay).convert('RGB')

def add_logo_layer(img, logo, position=(50, 50), scale=1.0, opacity=255, rotation=0, blend_mode="normal", grid_pieces=None):
    """Add logo/watermark with advanced blending options and grid-specific placement."""
    if logo.mode != 'RGBA':
        logo = logo.convert('RGBA')
    
    # Scale logo
    if scale != 1.0:
        new_size = (int(logo.size[0] * scale), int(logo.size[1] * scale))
        logo = logo.resize(new_size, Image.Resampling.LANCZOS)
    
    # Apply rotation
    if rotation != 0:
        logo = logo.rotate(rotation, expand=True)
    
    # Apply opacity
    if opacity < 255:
        alpha = logo.split()[-1]
        alpha = alpha.point(lambda p: int(p * (opacity / 255)))
        logo.putalpha(alpha)
    
    # Create composite
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Create overlay
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    
    # Paste logo at position
    if position[0] + logo.size[0] <= img.size[0] and position[1] + logo.size[1] <= img.size[1]:
        overlay.paste(logo, position, logo)
    
    # Apply blending
    if blend_mode == "multiply":
        result = Image.blend(img, overlay, 0.5)
    elif blend_mode == "screen":
        result = Image.alpha_composite(img, overlay)
    else:  # normal
        result = Image.alpha_composite(img, overlay)
    
    return result.convert('RGB')

def add_image_layer(base_img, layer_img, position=(50, 50), scale=1.0, opacity=255, rotation=0, blend_mode="normal", grid_pieces=None):
    """Add image layer with advanced blending options and grid-specific placement."""
    if layer_img.mode != 'RGBA':
        layer_img = layer_img.convert('RGBA')
    
    # Scale image
    if scale != 1.0:
        new_size = (int(layer_img.size[0] * scale), int(layer_img.size[1] * scale))
        layer_img = layer_img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Apply rotation
    if rotation != 0:
        layer_img = layer_img.rotate(rotation, expand=True)
    
    # Apply opacity
    if opacity < 255:
        alpha = layer_img.split()[-1]
        alpha = alpha.point(lambda p: int(p * (opacity / 255)))
        layer_img.putalpha(alpha)
    
    # Create composite
    if base_img.mode != 'RGBA':
        base_img = base_img.convert('RGBA')
    
    # Create overlay
    overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    
    # Paste image at position
    if position[0] + layer_img.size[0] <= base_img.size[0] and position[1] + layer_img.size[1] <= base_img.size[1]:
        overlay.paste(layer_img, position, layer_img)
    
    # Apply blending
    if blend_mode == "multiply":
        result = Image.blend(base_img, overlay, 0.5)
    elif blend_mode == "screen":
        result = Image.alpha_composite(base_img, overlay)
    else:  # normal
        result = Image.alpha_composite(base_img, overlay)
    
    return result.convert('RGB')

def get_number_style_config(style):
    """Enhanced number styling configurations."""
    configs = {
        "Red bold": {"fill": (255, 0, 0, 255), "outline": None, "outline_width": 0, "shadow": False},
        "White with black outline": {"fill": (255, 255, 255, 255), "outline": (0, 0, 0, 255), "outline_width": 4, "shadow": True},
        "Subtle gray": {"fill": (128, 128, 128, 200), "outline": None, "outline_width": 0, "shadow": False},
        "Colorful": {"fill": (255, 215, 0, 255), "outline": (255, 0, 255, 255), "outline_width": 3, "shadow": True},
        "Neon glow": {"fill": (0, 255, 255, 255), "outline": (0, 0, 255, 180), "outline_width": 6, "shadow": True},
        "Vintage badge": {"fill": (139, 69, 19, 255), "outline": (255, 255, 255, 255), "outline_width": 2, "shadow": True}
    }
    return configs.get(style, configs["Red bold"])

def add_number_overlay(img, number, style="Red bold", position="center"):
    """Enhanced number overlay with advanced styling."""
    overlay = img.copy().convert('RGBA')
    draw = ImageDraw.Draw(overlay)
    
    # Dynamic font size based on image size
    font_size = max(40, min(img.size[0] // 8, img.size[1] // 8))
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text = str(number)
    style_config = get_number_style_config(style)
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Enhanced position calculation
    margin = 25
    positions = {
        "center": ((img.size[0] - text_w) // 2, (img.size[1] - text_h) // 2),
        "top-left": (margin, margin),
        "top-right": (img.size[0] - text_w - margin, margin),
        "bottom-left": (margin, img.size[1] - text_h - margin),
        "bottom-right": (img.size[0] - text_w - margin, img.size[1] - text_h - margin)
    }
    
    x, y = positions.get(position, positions["center"])
    
    # Add shadow for vintage badge style
    if style_config.get("shadow"):
        shadow_offset = 3
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 100))
    
    # Draw outline if specified
    if style_config["outline"] and style_config["outline_width"] > 0:
        outline_width = style_config["outline_width"]
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx*dx + dy*dy <= outline_width*outline_width:
                    draw.text((x + dx, y + dy), text, font=font, fill=style_config["outline"])
    
    # Draw main text
    draw.text((x, y), text, font=font, fill=style_config["fill"])
    
    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

def split_image(img, rows, cols, add_spacing=False, spacing_width=0):
    """Enhanced image splitting with precise calculations."""
    w, h = img.size
    
    if add_spacing and spacing_width > 0:
        total_spacing_w = spacing_width * (cols - 1)
        total_spacing_h = spacing_width * (rows - 1)
        tile_w = (w - total_spacing_w) // cols
        tile_h = (h - total_spacing_h) // rows
    else:
        tile_w, tile_h = w // cols, h // rows
    
    pieces = []
    for r in range(rows):
        for c in range(cols):
            if add_spacing and spacing_width > 0:
                left = c * (tile_w + spacing_width)
                top = r * (tile_h + spacing_width)
            else:
                left, top = c * tile_w, r * tile_h
            
            right, bottom = left + tile_w, top + tile_h
            
            # Ensure we don't exceed image bounds
            right = min(right, w)
            bottom = min(bottom, h)
            
            piece = img.crop((left, top, right, bottom))
            pieces.append(piece)
    
    return pieces

def create_collage_preview(pieces, rows, cols, spacing=8):
    """Enhanced collage preview with better spacing."""
    if not pieces:
        return None
    
    piece_w, piece_h = pieces[0].size
    collage_w = cols * piece_w + (cols - 1) * spacing
    collage_h = rows * piece_h + (rows - 1) * spacing
    
    # Create background with subtle gradient
    collage = Image.new('RGB', (collage_w, collage_h), '#f0f0f0')
    
    for i, piece in enumerate(pieces):
        if i >= len(pieces):
            break
        row = i // cols
        col = i % cols
        x = col * (piece_w + spacing)
        y = row * (piece_h + spacing)
        collage.paste(piece, (x, y))
    
    return collage

def analyze_image_composition(img):
    """Analyze image composition for optimal text/logo placement."""
    # Convert to HSV for better analysis
    hsv = img.convert('HSV')
    h_channel = np.array(hsv.split()[2])  # Value channel
    
    # Find regions with low detail (good for text placement)
    blur_analysis = np.array(img.convert('L').filter(ImageFilter.GaussianBlur(radius=5)))
    detail_map = np.abs(np.array(img.convert('L')) - blur_analysis)
    
    # Suggest optimal text placement zones
    height, width = detail_map.shape
    zones = {
        'top-left': np.mean(detail_map[:height//3, :width//3]),
        'top-right': np.mean(detail_map[:height//3, 2*width//3:]),
        'bottom-left': np.mean(detail_map[2*height//3:, :width//3]),
        'bottom-right': np.mean(detail_map[2*height//3:, 2*width//3:]),
        'center': np.mean(detail_map[height//3:2*height//3, width//3:2*width//3])
    }
    
    best_zone = min(zones.items(), key=lambda x: x[1])
    return best_zone[0], zones

def generate_enhanced_posting_guide(rows, cols, grid_type="feed", upload_schedule="optimal"):
    """Generate comprehensive posting instructions with timing."""
    total_posts = rows * cols
    
    if grid_type == "carousel":
        guide = f"""
        **📱 CAROUSEL POSTING STRATEGY:**
        
        **Upload Order:** Sequential (1 → {total_posts})
        ✅ Drag files in numerical order when creating carousel
        ✅ First image becomes cover - choose wisely!
        ✅ Perfect for storytelling and product details
        
        **Engagement Tips:**
        • Add "Swipe for more" in caption
        • Use consistent visual style across slides
        • End with call-to-action slide
        """
    else:
        guide = f"""
        **📋 GRID FEED POSTING STRATEGY:**
        
        **Critical Upload Order:** REVERSE (Start with #{total_posts})
        
        **Step-by-Step Process:**
        1. Post IG_{total_posts:02d}.jpg FIRST
        2. Wait {15 if upload_schedule == 'fast' else 30 if upload_schedule == 'optimal' else 60} minutes
        3. Post IG_{total_posts-1:02d}.jpg
        4. Continue in reverse order...
        5. Post IG_01.jpg LAST
        
        **Why Reverse Order?**
        Instagram feed displays newest posts first, so posting in reverse 
        ensures your grid appears correctly aligned for visitors.
        
        **Timing Strategy:**
        • Peak hours: 11 AM - 1 PM and 7 PM - 9 PM
        • Avoid posting all at once (algorithm penalty)
        • Use Instagram scheduling tools for precision
        """
    
    return guide

def calculate_grid_statistics(pieces, original_size):
    """Calculate comprehensive statistics about the grid."""
    total_pixels = sum(piece.size[0] * piece.size[1] for piece in pieces)
    original_pixels = original_size[0] * original_size[1]
    
    # Calculate compression ratio
    compression_ratio = total_pixels / original_pixels if original_pixels > 0 else 1
    
    # Average piece size
    avg_width = sum(piece.size[0] for piece in pieces) / len(pieces)
    avg_height = sum(piece.size[1] for piece in pieces) / len(pieces)
    
    return {
        'total_pieces': len(pieces),
        'total_pixels': total_pixels,
        'compression_ratio': compression_ratio,
        'avg_dimensions': (avg_width, avg_height),
        'grid_efficiency': compression_ratio * 100
    }

# --- Advanced Layer Management System ---
def add_layer(layer_type, layer_data):
    """Add a new layer to the session state."""
    st.session_state.layer_counter += 1
    layer = {
        'id': st.session_state.layer_counter,
        'type': layer_type,
        'data': layer_data,
        'visible': True,
        'opacity': 1.0,
        'z_index': st.session_state.layer_counter,
        'grid_pieces': layer_data.get('grid_pieces', [])  # Which grid pieces this layer applies to
    }
    st.session_state.layers.append(layer)

def remove_layer(layer_id):
    """Remove a layer by ID."""
    st.session_state.layers = [layer for layer in st.session_state.layers if layer['id'] != layer_id]

def move_layer_up(layer_id):
    """Move layer up in z-order."""
    layers = st.session_state.layers
    for i, layer in enumerate(layers):
        if layer['id'] == layer_id and i < len(layers) - 1:
            layers[i], layers[i + 1] = layers[i + 1], layers[i]
            break

def move_layer_down(layer_id):
    """Move layer down in z-order."""
    layers = st.session_state.layers
    for i, layer in enumerate(layers):
        if layer['id'] == layer_id and i > 0:
            layers[i], layers[i - 1] = layers[i - 1], layers[i]
            break

def move_layer_to_front(layer_id):
    """Move layer to front."""
    layers = st.session_state.layers
    layer_to_move = None
    for i, layer in enumerate(layers):
        if layer['id'] == layer_id:
            layer_to_move = layers.pop(i)
            break
    if layer_to_move:
        layers.append(layer_to_move)

def move_layer_to_back(layer_id):
    """Move layer to back."""
    layers = st.session_state.layers
    layer_to_move = None
    for i, layer in enumerate(layers):
        if layer['id'] == layer_id:
            layer_to_move = layers.pop(i)
            break
    if layer_to_move:
        layers.insert(0, layer_to_move)

def update_layer_position(layer_id, new_x, new_y):
    """Update layer position."""
    for layer in st.session_state.layers:
        if layer['id'] == layer_id:
            if layer['type'] in ['text', 'logo', 'image']:
                layer['data']['position'] = (new_x, new_y)
            break

def apply_all_layers(base_image, grid_piece_index=None):
    """Apply all layers to the base image, optionally for a specific grid piece."""
    result = base_image.copy()
    
    # Sort layers by z-index
    sorted_layers = sorted(st.session_state.layers, key=lambda x: x.get('z_index', 0))
    
    for layer in sorted_layers:
        if not layer['visible']:
            continue
        
        # Check if this layer applies to the current grid piece
        if grid_piece_index is not None and layer['grid_pieces']:
            if grid_piece_index not in layer['grid_pieces']:
                continue
            
        if layer['type'] == 'text':
            data = layer['data']
            result = add_text_layer(
                result, 
                data['text'],
                data['font_size'],
                data['color'],
                data['position'],
                data['font_style'],
                data['rotation'],
                int(data['opacity'] * layer['opacity'] * 255),
                data.get('outline_color'),
                data.get('outline_width', 0),
                data.get('shadow', False),
                data.get('shadow_offset', (5, 5)),
                data.get('shadow_color', (0, 0, 0)),
                data.get('grid_pieces', [])
            )
        elif layer['type'] == 'logo':
            data = layer['data']
            result = add_logo_layer(
                result,
                data['logo'],
                data['position'],
                data['scale'],
                int(data['opacity'] * layer['opacity'] * 255),
                data['rotation'],
                data['blend_mode'],
                data.get('grid_pieces', [])
            )
        elif layer['type'] == 'image':
            data = layer['data']
            result = add_image_layer(
                result,
                data['image'],
                data['position'],
                data['scale'],
                int(data['opacity'] * layer['opacity'] * 255),
                data['rotation'],
                data['blend_mode'],
                data.get('grid_pieces', [])
            )
    
    return result

# --- Main Application Logic ---
if uploaded_file:
    # Load and display original image info
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size
    original_format = uploaded_file.type
    
    # Enhanced statistics display
    st.subheader("📊 Image Analysis")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <h4>📏 Dimensions</h4>
            <p>{original_size[0]} × {original_size[1]} px</p>
            <small>{original_size[0] * original_size[1] / 1000000:.1f} MP</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        file_size = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div class="stats-box">
            <h4>📦 File Size</h4>
            <p>{file_size:.1f} KB</p>
            <small>{file_size/1024:.2f} MB</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        aspect_ratio = original_size[0] / original_size[1]
        ratio_text = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Landscape" if aspect_ratio > 1 else "Portrait"
        st.markdown(f"""
        <div class="stats-box">
            <h4>📐 Aspect Ratio</h4>
            <p>{aspect_ratio:.2f}:1</p>
            <small>{ratio_text}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Color analysis
        colors = image.getcolors(maxcolors=256*256*256)
        color_count = len(colors) if colors else "Many"
        st.markdown(f"""
        <div class="stats-box">
            <h4>🎨 Colors</h4>
            <p>{color_count}</p>
            <small>{original_format}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        # Composition analysis
        best_zone, zones = analyze_image_composition(image)
        st.markdown(f"""
        <div class="stats-box">
            <h4>🎯 Best Text Zone</h4>
            <p>{best_zone.replace('-', ' ').title()}</p>
            <small>For overlays</small>
        </div>
        """, unsafe_allow_html=True)

    # Get grid dimensions
    if "Custom" in grid_option:
        rows, cols = custom_rows, custom_cols
    else:
        rows, cols = get_grid_dimensions(grid_option)

    # --- Grid Piece Selection ---
    st.subheader("🎯 Grid Piece Selection")
    st.write("**Select which grid pieces to apply layers to (leave empty for all pieces):**")
    
    grid_cols = st.columns(min(cols, 6))
    total_pieces = rows * cols
    
    for i in range(total_pieces):
        with grid_cols[i % len(grid_cols)]:
            piece_selected = st.checkbox(f"Piece {i+1}", key=f"grid_piece_{i}")
            if piece_selected and i not in st.session_state.selected_grid_pieces:
                st.session_state.selected_grid_pieces.append(i)
            elif not piece_selected and i in st.session_state.selected_grid_pieces:
                st.session_state.selected_grid_pieces.remove(i)

    # --- Advanced Layer Management Section ---
    st.subheader("🎨 Advanced Layer Management System")
    
    layer_col1, layer_col2, layer_col3 = st.columns([2, 2, 2])
    
    with layer_col1:
        st.markdown('<div class="layer-panel">', unsafe_allow_html=True)
        st.write("**📝 Add Text Layer**")
        
        text_input = st.text_input("Text content", placeholder="Enter your text here...")
        
        text_col1, text_col2 = st.columns(2)
        with text_col1:
            text_font_size = st.slider("Font size", 10, 200, 60, key="text_font")
            text_color = st.color_picker("Text color", "#FFFFFF", key="text_color")
        with text_col2:
            text_position_x = st.slider("X Position", 0, original_size[0], original_size[0]//4, key="text_x")
            text_position_y = st.slider("Y Position", 0, original_size[1], original_size[1]//4, key="text_y")
        
        text_style_col1, text_style_col2 = st.columns(2)
        with text_style_col1:
            text_font_style = st.selectbox("Font style", ["bold", "italic", "normal"], key="text_style")
            text_rotation = st.slider("Rotation", -180, 180, 0, key="text_rotation")
        with text_style_col2:
            text_opacity = st.slider("Opacity", 0.0, 1.0, 1.0, 0.1, key="text_opacity")
            text_shadow = st.checkbox("Add shadow", key="text_shadow")
        
        # Advanced text options
        with st.expander("Advanced Text Options"):
            text_outline = st.checkbox("Add outline", key="text_outline")
            if text_outline:
                text_outline_color = st.color_picker("Outline color", "#000000", key="text_outline_color")
                text_outline_width = st.slider("Outline width", 1, 10, 2, key="text_outline_width")
            else:
                text_outline_color, text_outline_width = None, 0
        
        if st.button("➕ Add Text Layer", disabled=not text_input):
            # Convert hex color to RGB
            text_rgb = tuple(int(text_color[i:i+2], 16) for i in (1, 3, 5))
            outline_rgb = tuple(int(text_outline_color[i:i+2], 16) for i in (1, 3, 5)) if text_outline_color else None
            
            text_layer_data = {
                'text': text_input,
                'font_size': text_font_size,
                'color': text_rgb,
                'position': (text_position_x, text_position_y),
                'font_style': text_font_style,
                'rotation': text_rotation,
                'opacity': text_opacity,
                'outline_color': outline_rgb,
                'outline_width': text_outline_width,
                'shadow': text_shadow,
                'shadow_offset': (5, 5),
                'shadow_color': (0, 0, 0),
                'grid_pieces': st.session_state.selected_grid_pieces.copy()
            }
            add_layer('text', text_layer_data)
            st.success(f"✅ Text layer '{text_input}' added!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with layer_col2:
        st.markdown('<div class="layer-panel">', unsafe_allow_html=True)
        st.write("**🖼️ Add Image Layer**")
        
        if additional_images:
            selected_image_idx = st.selectbox("Select image", range(len(additional_images)), 
                                            format_func=lambda x: additional_images[x].name)
            
            if selected_image_idx is not None:
                selected_image = Image.open(additional_images[selected_image_idx])
                st.image(selected_image, caption="Selected Image", width=150)
                
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    img_scale = st.slider("Image scale", 0.1, 3.0, 0.5, 0.1, key="img_scale")
                    img_position_x = st.slider("X Position", 0, original_size[0], original_size[0]//10, key="img_x")
                with img_col2:
                    img_opacity = st.slider("Image opacity", 0.0, 1.0, 1.0, 0.1, key="img_opacity")
                    img_position_y = st.slider("Y Position", 0, original_size[1], original_size[1]//10, key="img_y")
                
                img_rotation = st.slider("Image rotation", -180, 180, 0, key="img_rotation")
                img_blend_mode = st.selectbox("Blend mode", ["normal", "multiply", "screen"], key="img_blend")
                
                # Image editing options
                with st.expander("Image Editing Options"):
                    resize_factor = st.slider("Resize Factor", 0.1, 3.0, 1.0, 0.1, key="img_resize")
                    crop_enabled = st.checkbox("Enable Cropping", key="img_crop_enabled")
                    if crop_enabled:
                        crop_left = st.slider("Crop Left", 0, selected_image.size[0]//2, 0, key="crop_left")
                        crop_top = st.slider("Crop Top", 0, selected_image.size[1]//2, 0, key="crop_top")
                        crop_right = st.slider("Crop Right", selected_image.size[0]//2, selected_image.size[0], selected_image.size[0], key="crop_right")
                        crop_bottom = st.slider("Crop Bottom", selected_image.size[1]//2, selected_image.size[1], selected_image.size[1], key="crop_bottom")
                
                if st.button("➕ Add Image Layer"):
                    # Apply image editing
                    processed_image = selected_image.copy()
                    
                    if crop_enabled:
                        processed_image = crop_image_layer(processed_image, (crop_left, crop_top, crop_right, crop_bottom))
                    
                    if resize_factor != 1.0:
                        processed_image = resize_image_layer(processed_image, resize_factor)
                    
                    img_layer_data = {
                        'image': processed_image,
                        'position': (img_position_x, img_position_y),
                        'scale': img_scale,
                        'opacity': img_opacity,
                        'rotation': img_rotation,
                        'blend_mode': img_blend_mode,
                        'grid_pieces': st.session_state.selected_grid_pieces.copy()
                    }
                    add_layer('image', img_layer_data)
                    st.success("✅ Image layer added!")
        else:
            st.info("Upload additional images in the sidebar to add image layers")
        
        # Logo layer section
        st.write("**🏷️ Add Logo Layer**")
        if logo_file:
            logo_image = Image.open(logo_file)
            st.image(logo_image, caption="Logo Preview", width=150)
            
            logo_col1, logo_col2 = st.columns(2)
            with logo_col1:
                logo_scale = st.slider("Logo scale", 0.1, 3.0, 0.3, 0.1, key="logo_scale")
                logo_position_x = st.slider("X Position", 0, original_size[0], original_size[0]//10, key="logo_x")
            with logo_col2:
                logo_opacity = st.slider("Logo opacity", 0.0, 1.0, 0.8, 0.1, key="logo_opacity")
                logo_position_y = st.slider("Y Position", 0, original_size[1], original_size[1]//10, key="logo_y")
            
            logo_rotation = st.slider("Logo rotation", -180, 180, 0, key="logo_rotation")
            logo_blend_mode = st.selectbox("Blend mode", ["normal", "multiply", "screen"], key="logo_blend")
            
            if st.button("➕ Add Logo Layer"):
                logo_layer_data = {
                    'logo': logo_image,
                    'position': (logo_position_x, logo_position_y),
                    'scale': logo_scale,
                    'opacity': logo_opacity,
                    'rotation': logo_rotation,
                    'blend_mode': logo_blend_mode,
                    'grid_pieces': st.session_state.selected_grid_pieces.copy()
                }
                add_layer('logo', logo_layer_data)
                st.success("✅ Logo layer added!")
        else:
            st.info("Upload a logo file to add logo layers")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with layer_col3:
        st.markdown('<div class="layer-panel">', unsafe_allow_html=True)
        st.write("**📋 Advanced Layer List**")
        
        if st.session_state.layers:
            # Sort layers by z-index for display
            sorted_layers = sorted(st.session_state.layers, key=lambda x: x.get('z_index', 0), reverse=True)
            
            for layer in sorted_layers:
                layer_name = f"{layer['type'].title()} {layer['id']}"
                if layer['type'] == 'text':
                    layer_name += f": {layer['data']['text'][:10]}..."
                elif layer['type'] == 'image':
                    layer_name += f": Image Layer"
                
                st.markdown(f'<div class="layer-item">', unsafe_allow_html=True)
                
                # Layer visibility and name
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    layer['visible'] = st.checkbox(layer_name, value=layer['visible'], key=f"layer_vis_{layer['id']}")
                with col_b:
                    if st.button("🗑️", key=f"del_{layer['id']}", help="Delete layer"):
                        remove_layer(layer['id'])
                        st.rerun()
                
                if layer['visible']:
                    # Layer opacity
                    layer['opacity'] = st.slider(
                        "Opacity", 0.0, 1.0, layer['opacity'], 0.1, 
                        key=f"layer_opacity_{layer['id']}"
                    )
                    
                    # Layer controls
                    st.markdown('<div class="layer-controls">', unsafe_allow_html=True)
                    
                    # Position controls
                    pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
                    with pos_col1:
                        if st.button("⬅️", key=f"left_{layer['id']}", help="Move left"):
                            current_pos = layer['data']['position']
                            update_layer_position(layer['id'], max(0, current_pos[0] - 10), current_pos[1])
                            st.rerun()
                    with pos_col2:
                        if st.button("➡️", key=f"right_{layer['id']}", help="Move right"):
                            current_pos = layer['data']['position']
                            update_layer_position(layer['id'], min(original_size[0], current_pos[0] + 10), current_pos[1])
                            st.rerun()
                    with pos_col3:
                        if st.button("⬆️", key=f"up_{layer['id']}", help="Move up"):
                            current_pos = layer['data']['position']
                            update_layer_position(layer['id'], current_pos[0], max(0, current_pos[1] - 10))
                            st.rerun()
                    with pos_col4:
                        if st.button("⬇️", key=f"down_{layer['id']}", help="Move down"):
                            current_pos = layer['data']['position']
                            update_layer_position(layer['id'], current_pos[0], min(original_size[1], current_pos[1] + 10))
                            st.rerun()
                    
                    # Z-order controls
                    z_col1, z_col2, z_col3, z_col4 = st.columns(4)
                    with z_col1:
                        if st.button("🔼", key=f"z_up_{layer['id']}", help="Move layer up"):
                            move_layer_up(layer['id'])
                            st.rerun()
                    with z_col2:
                        if st.button("🔽", key=f"z_down_{layer['id']}", help="Move layer down"):
                            move_layer_down(layer['id'])
                            st.rerun()
                    with z_col3:
                        if st.button("⏫", key=f"z_front_{layer['id']}", help="Bring to front"):
                            move_layer_to_front(layer['id'])
                            st.rerun()
                    with z_col4:
                        if st.button("⏬", key=f"z_back_{layer['id']}", help="Send to back"):
                            move_layer_to_back(layer['id'])
                            st.rerun()
                    
                    # Edit button for text layers
                    if layer['type'] == 'text':
                        if st.button("✏️ Edit Text", key=f"edit_{layer['id']}"):
                            st.session_state.editing_layer_id = layer['id']
                            st.rerun()
                    
                    # Show grid pieces this layer applies to
                    if layer['grid_pieces']:
                        st.caption(f"Applied to pieces: {', '.join(map(str, [p+1 for p in layer['grid_pieces']]))}")
                    else:
                        st.caption("Applied to all pieces")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No layers added yet")
        
        # Layer management buttons
        layer_mgmt_col1, layer_mgmt_col2 = st.columns(2)
        with layer_mgmt_col1:
            if st.button("🗑️ Clear All Layers"):
                st.session_state.layers = []
                st.rerun()
        with layer_mgmt_col2:
            if st.button("📋 Duplicate Selected"):
                # Duplicate the last selected layer
                if st.session_state.layers:
                    last_layer = st.session_state.layers[-1].copy()
                    last_layer['id'] = st.session_state.layer_counter + 1
                    st.session_state.layer_counter += 1
                    st.session_state.layers.append(last_layer)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Text editing modal
    if st.session_state.editing_layer_id:
        editing_layer = next((layer for layer in st.session_state.layers if layer['id'] == st.session_state.editing_layer_id), None)
        if editing_layer and editing_layer['type'] == 'text':
            st.subheader("✏️ Edit Text Layer")
            
            edit_col1, edit_col2, edit_col3 = st.columns([2, 2, 1])
            
            with edit_col1:
                new_text = st.text_input("Edit text:", value=editing_layer['data']['text'], key="edit_text_input")
                new_font_size = st.slider("Font size:", 10, 200, editing_layer['data']['font_size'], key="edit_font_size")
            
            with edit_col2:
                new_color = st.color_picker("Text color:", f"#{editing_layer['data']['color'][0]:02x}{editing_layer['data']['color'][1]:02x}{editing_layer['data']['color'][2]:02x}", key="edit_color")
                new_rotation = st.slider("Rotation:", -180, 180, editing_layer['data']['rotation'], key="edit_rotation")
            
            with edit_col3:
                if st.button("💾 Save Changes"):
                    editing_layer['data']['text'] = new_text
                    editing_layer['data']['font_size'] = new_font_size
                    editing_layer['data']['color'] = tuple(int(new_color[i:i+2], 16) for i in (1, 3, 5))
                    editing_layer['data']['rotation'] = new_rotation
                    st.session_state.editing_layer_id = None
                    st.success("✅ Text layer updated!")
                    st.rerun()
                
                if st.button("❌ Cancel"):
                    st.session_state.editing_layer_id = None
                    st.rerun()

    # Apply image enhancements
    enhanced_image = enhance_image(
        image, 
        enhance_brightness, 
        enhance_contrast, 
        enhance_saturation, 
        enhance_sharpness
    )
    
    # Apply color grading if enabled
    if apply_color_grade:
        enhanced_image = apply_color_grading(
            enhanced_image,
            temperature,
            tint,
            highlights,
            shadows
        )
    
    # Apply advanced filters
    filtered_image = apply_advanced_filters(
        enhanced_image,
        blur_radius,
        vintage_strength,
        vignette_strength
    )
    
    # Apply layers
    layered_image = apply_all_layers(filtered_image)
    
    # Apply square formatting
    processed_image = make_square(
        layered_image, 
        option=square_option, 
        size=output_size, 
        pad_color=pad_color
    )

    # --- Enhanced Image Comparison ---
    st.subheader("🖼️ Processing Stages Preview")
    
    processing_tabs = st.tabs(["📷 Original", "✨ Enhanced", "🎭 Filtered", "🎨 With Layers", "📐 Final"])
    
    with processing_tabs[0]:
        st.image(image, caption="Original Image", use_container_width=True)
        
    with processing_tabs[1]:
        st.image(enhanced_image, caption="Enhanced (Brightness, Contrast, etc.)", use_container_width=True)
        
    with processing_tabs[2]:
        st.image(filtered_image, caption="With Advanced Filters", use_container_width=True)
        
    with processing_tabs[3]:
        st.image(layered_image, caption="With Text & Logo Layers", use_container_width=True)
        
    with processing_tabs[4]:
        st.image(processed_image, caption=f"Final Processed ({square_option})", use_container_width=True)

    # Split image into pieces
    pieces = split_image(processed_image, rows, cols)
    
    # Apply layers to individual pieces if specified
    final_pieces = []
    for i, piece in enumerate(pieces):
        piece_with_layers = apply_all_layers(piece, grid_piece_index=i)
        final_pieces.append(piece_with_layers)
    
    # Calculate grid statistics
    grid_stats = calculate_grid_statistics(final_pieces, original_size)

    # --- Enhanced Grid Preview ---
    st.subheader(f"✨ Grid Preview ({rows} × {cols} = {len(final_pieces)} pieces)")
    
    # Grid statistics
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Total Pieces", grid_stats['total_pieces'])
    with stats_col2:
        st.metric("Avg Piece Size", f"{grid_stats['avg_dimensions'][0]:.0f}×{grid_stats['avg_dimensions'][1]:.0f}")
    with stats_col3:
        st.metric("Grid Efficiency", f"{grid_stats['grid_efficiency']:.1f}%")
    with stats_col4:
        est_posts_time = grid_stats['total_pieces'] * 30  # 30 min intervals
        st.metric("Est. Posting Time", f"{est_posts_time//60:.0f}h {est_posts_time%60}m")

    # Create tabs for different preview modes
    preview_tabs = st.tabs([
        "🔢 Numbered Preview", 
        "🖼️ Clean Preview", 
        "📱 Collage View", 
        "🎯 Posting Order",
        "📊 Quality Analysis"
    ])
    
    with preview_tabs[0]:
        st.markdown('<div class="preview-grid">', unsafe_allow_html=True)
        preview_cols = st.columns(cols)
        
        for idx, piece in enumerate(final_pieces):
            if add_numbers:
                numbered_piece = add_number_overlay(piece, idx + 1, number_style, number_position)
            else:
                numbered_piece = piece
            
            with preview_cols[idx % cols]:
                st.image(numbered_piece, use_container_width=True, caption=f"Part {idx+1}")
                
                # Show piece statistics
                piece_size = piece.size[0] * piece.size[1] / 1000
                st.caption(f"📐 {piece.size[0]}×{piece.size[1]} ({piece_size:.0f}K pixels)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with preview_tabs[1]:
        st.markdown('<div class="preview-grid">', unsafe_allow_html=True)
        clean_cols = st.columns(cols)
        for idx, piece in enumerate(final_pieces):
            with clean_cols[idx % cols]:
                st.image(piece, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with preview_tabs[2]:
        if create_collage:
            collage = create_collage_preview(final_pieces, rows, cols, spacing=12)
            if collage:
                st.image(collage, caption="Complete Grid Collage Preview", use_container_width=True)
                
                # Collage download option
                collage_buffer = io.BytesIO()
                collage.save(collage_buffer, format="PNG", optimize=True)
                st.download_button(
                    "📸 Download Collage Preview",
                    data=collage_buffer.getvalue(),
                    file_name=f"grid_collage_{rows}x{cols}.png",
                    mime="image/png"
                )
        else:
            st.info("Enable 'Create collage preview' in sidebar to see full grid view")
    
    with preview_tabs[3]:
        st.write("**📋 Optimal Posting Order for Instagram Feed Grid:**")
        st.warning("⚠️ **IMPORTANT:** Post in REVERSE order for proper grid alignment!")
        
        # Visual posting order guide
        order_cols = st.columns(min(cols, 4))
        posting_order = list(range(len(final_pieces), 0, -1))
        
        for idx in range(len(final_pieces)):
            row = idx // cols
            col = idx % cols
            with order_cols[col % len(order_cols)]:
                post_num = posting_order[idx]
                file_name = f"IG_{idx+1:02d}"
                
                # Color code by posting priority
                if post_num <= 3:
                    st.error(f"🔴 POST #{post_num} FIRST\n{file_name}")
                elif post_num <= len(final_pieces) // 2:
                    st.warning(f"🟡 Post #{post_num}\n{file_name}")
                else:
                    st.success(f"🟢 Post #{post_num} Last\n{file_name}")
        
        # Posting schedule calculator
        st.write("**⏰ Posting Schedule Calculator:**")
        schedule_interval = st.selectbox(
            "Posting interval:",
            ["15 minutes", "30 minutes", "1 hour", "2 hours", "Daily"],
            index=1
        )
        
        start_time = st.time_input("Start posting at:", value=datetime.now().time())
        
        # Calculate posting schedule
        intervals = {"15 minutes": 15, "30 minutes": 30, "1 hour": 60, "2 hours": 120, "Daily": 1440}
        interval_mins = intervals[schedule_interval]
        
        st.write("**📅 Your Posting Schedule:**")
        for i, post_order in enumerate(posting_order):
            post_time = datetime.combine(datetime.today(), start_time) + timedelta(minutes=i * interval_mins)
            file_name = f"IG_{list(range(1, len(final_pieces) + 1))[post_order - 1]:02d}"
            st.write(f"**Post #{post_order}:** {file_name} at {post_time.strftime('%I:%M %p on %b %d')}")
    
    with preview_tabs[4]:
        st.write("**🔍 Quality Analysis Report:**")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            # Resolution analysis
            piece_resolution = final_pieces[0].size[0] * final_pieces[0].size[1]
            resolution_quality = "Excellent" if piece_resolution >= 1000000 else "Good" if piece_resolution >= 500000 else "Fair"
            
            st.metric("Resolution Quality", resolution_quality)
            st.metric("Pixels per piece", f"{piece_resolution:,}")
            
            # Compression analysis
            original_total = original_size[0] * original_size[1]
            pieces_total = sum(p.size[0] * p.size[1] for p in final_pieces)
            efficiency = (pieces_total / original_total) * 100
            st.metric("Grid Efficiency", f"{efficiency:.1f}%")
        
        with analysis_col2:
            # Color consistency analysis
            sample_piece = final_pieces[0]
            dominant_colors = sample_piece.getcolors(maxcolors=256*256*256)
            if dominant_colors:
                dominant_color = max(dominant_colors, key=lambda x: x[0])
                color_hex = f"#{dominant_color[1][0]:02x}{dominant_color[1][1]:02x}{dominant_color[1][2]:02x}"
                st.color_picker("Dominant Color", color_hex, disabled=True)
            
            # Contrast analysis
            gray_piece = sample_piece.convert('L')
            pixel_values = list(gray_piece.getdata())
            contrast_score = np.std(pixel_values) / 255 * 100
            contrast_quality = "High" if contrast_score > 25 else "Medium" if contrast_score > 15 else "Low"
            st.metric("Contrast Level", f"{contrast_quality} ({contrast_score:.1f}%)")

    # --- Enhanced Download Section with Cloud Integration ---
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.subheader("⬇️ Download Your Professional Grid")
    
    download_col1, download_col2, download_col3 = st.columns([2, 1, 1])
    
    with download_col1:
        # Enhanced ZIP creation with multiple formats
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            
            # Add grid pieces
            for i, piece in enumerate(final_pieces, 1):
                # PNG version
                if output_format in ["PNG", "Both"]:
                    png_bytes = io.BytesIO()
                    piece.save(png_bytes, format="PNG", optimize=True)
                    zf.writestr(f"PNG_Format/IG_{i:02d}.png", png_bytes.getvalue())
                
                # JPG version
                if output_format in ["JPG", "Both"]:
                    jpg_bytes = io.BytesIO()
                    piece.save(jpg_bytes, format="JPEG", quality=quality_setting, optimize=True)
                    zf.writestr(f"JPG_Format/IG_{i:02d}.jpg", jpg_bytes.getvalue())
            
            # Add collage if created
            if create_collage and 'collage' in locals():
                collage_bytes = io.BytesIO()
                collage.save(collage_bytes, format="PNG", optimize=True)
                zf.writestr("Extras/grid_collage_preview.png", collage_bytes.getvalue())
            
            # Add original processed image
            original_bytes = io.BytesIO()
            processed_image.save(original_bytes, format="PNG", optimize=True)
            zf.writestr("Extras/processed_original.png", original_bytes.getvalue())
            
            # Enhanced posting guide
            guide_content = generate_enhanced_posting_guide(rows, cols, "feed", "optimal")
            zf.writestr("POSTING_GUIDE.txt", guide_content)
            
            # Add layer information
            if st.session_state.layers:
                layer_info = "LAYER INFORMATION:\n\n"
                for layer in st.session_state.layers:
                    layer_info += f"Layer {layer['id']} ({layer['type']}):\n"
                    if layer['type'] == 'text':
                        layer_info += f"  Text: {layer['data']['text']}\n"
                        layer_info += f"  Position: {layer['data']['position']}\n"
                        layer_info += f"  Font Size: {layer['data']['font_size']}\n"
                    elif layer['type'] in ['logo', 'image']:
                        layer_info += f"  Scale: {layer['data']['scale']}\n"
                        layer_info += f"  Position: {layer['data']['position']}\n"
                    layer_info += f"  Opacity: {layer['opacity']}\n"
                    layer_info += f"  Visible: {layer['visible']}\n"
                    if layer['grid_pieces']:
                        layer_info += f"  Grid Pieces: {layer['grid_pieces']}\n"
                    layer_info += "\n"
                
                zf.writestr("LAYER_INFO.txt", layer_info)

        # Main download button
        zip_filename = f"instagram_grid_advanced_{rows}x{cols}_{uploaded_file.name.split('.')[0]}.zip"
        st.download_button(
            "📦 Download Complete Advanced Package",
            data=zip_buffer.getvalue(),
            file_name=zip_filename,
            mime="application/zip",
            help=f"Downloads all {len(final_pieces)} pieces, guides, and extras with advanced layer support"
        )
        
        # Google Drive Upload Option (same as before but with final_pieces)
        if st.session_state.google_authenticated and GOOGLE_APIS_AVAILABLE:
            st.write("**☁️ Cloud Upload Options:**")
            
            upload_to_drive_option = st.checkbox("📤 Upload to Google Drive", value=False)
            
            if upload_to_drive_option:
                drive_folder_name = st.text_input(
                    "Drive folder name:", 
                    value=f"Instagram_Grid_Advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                if st.button("🚀 Upload to Google Drive"):
                    with st.spinner("Uploading to Google Drive..."):
                        try:
                            # Create folder for this project
                            folder_id, success = create_drive_folder(
                                st.session_state.drive_service, 
                                drive_folder_name,
                                st.session_state.get('selected_folder_id')
                            )
                            
                            if success:
                                upload_count = 0
                                failed_uploads = []
                                
                                # Upload individual pieces
                                for i, piece in enumerate(final_pieces, 1):
                                    try:
                                        img_bytes = io.BytesIO()
                                        if output_format == "PNG":
                                            piece.save(img_bytes, format="PNG", optimize=True)
                                            filename = f"IG_{i:02d}.png"
                                            mime_type = "image/png"
                                        else:
                                            piece.save(img_bytes, format="JPEG", quality=quality_setting, optimize=True)
                                            filename = f"IG_{i:02d}.jpg"
                                            mime_type = "image/jpeg"
                                        
                                        file_id, web_link, upload_success = upload_to_drive(
                                            st.session_state.drive_service,
                                            img_bytes.getvalue(),
                                            filename,
                                            folder_id,
                                            mime_type
                                        )
                                        
                                        if upload_success:
                                            upload_count += 1
                                        else:
                                            failed_uploads.append(filename)
                                    
                                    except Exception as e:
                                        failed_uploads.append(f"{filename} (Error: {str(e)})")
                                
                                # Upload ZIP file
                                try:
                                    zip_file_id, zip_web_link, zip_success = upload_to_drive(
                                        st.session_state.drive_service,
                                        zip_buffer.getvalue(),
                                        zip_filename,
                                        folder_id,
                                        "application/zip"
                                    )
                                    
                                    if zip_success:
                                        upload_count += 1
                                
                                except Exception as e:
                                    failed_uploads.append(f"ZIP file (Error: {str(e)})")
                                
                                # Show results
                                if upload_count > 0:
                                    st.success(f"✅ Successfully uploaded {upload_count} files to Google Drive!")
                                    st.info(f"📁 Folder: {drive_folder_name}")
                                
                                if failed_uploads:
                                    st.warning(f"⚠️ Failed to upload {len(failed_uploads)} files:")
                                    for failed in failed_uploads:
                                        st.write(f"• {failed}")
                                
                                # Store project data for sheets
                                st.session_state.project_data = {
                                    'name': drive_folder_name,
                                    'grid_layout': f"{rows}×{cols}",
                                    'total_pieces': len(final_pieces),
                                    'output_size': output_size,
                                    'quality': quality_setting,
                                    'layers_count': len(st.session_state.layers),
                                    'drive_folder_id': folder_id,
                                    'notes': f"Advanced version with {len(st.session_state.layers)} layers, uploaded {upload_count} files successfully"
                                }
                            
                            else:
                                st.error("❌ Failed to create Google Drive folder")
                        
                        except Exception as e:
                            st.error(f"❌ Upload failed: {str(e)}")
            
            # Google Sheets Integration (same as before)
            if 'project_data' in st.session_state and st.session_state.project_data:
                st.write("**📊 Project Tracking:**")
                
                create_sheet_option = st.checkbox("📈 Create project tracking sheet", value=True)
                
                if create_sheet_option and st.button("📊 Create Google Sheet"):
                    with st.spinner("Creating Google Sheet..."):
                        try:
                            sheet_name = f"Instagram_Projects_Advanced_{datetime.now().strftime('%Y%m%d')}"
                            sheet_id, sheet_url, sheet_success = create_project_sheet(
                                st.session_state.sheets_client,
                                sheet_name,
                                st.session_state.project_data
                            )
                            
                            if sheet_success:
                                st.success("✅ Google Sheet created successfully!")
                                st.markdown(f"🔗 [Open Sheet]({sheet_url})")
                            else:
                                st.error("❌ Failed to create Google Sheet")
                        
                        except Exception as e:
                            st.error(f"❌ Sheet creation failed: {str(e)}")
    
    with download_col2:
        # Enhanced statistics
        total_size_mb = sum(len(piece.tobytes()) for piece in final_pieces) / 1024 / 1024
        st.metric("Package Size", f"{total_size_mb:.2f} MB")
        st.metric("Pieces", len(final_pieces))
        st.metric("Layers Applied", len([l for l in st.session_state.layers if l['visible']]))
        
        # Quality score calculation
        quality_factors = [
            min(100, (final_pieces[0].size[0] / 1080) * 100),  # Resolution score
            contrast_score if 'contrast_score' in locals() else 50,  # Contrast score
            (quality_setting / 100) * 100,  # Quality setting score
        ]
        overall_quality = sum(quality_factors) / len(quality_factors)
        quality_grade = "A+" if overall_quality >= 90 else "A" if overall_quality >= 80 else "B+" if overall_quality >= 70 else "B"
        st.metric("Quality Grade", quality_grade)
        
        # Advanced features status
        st.metric("Advanced Features", "✅ Active")
        
        # Cloud status
        if st.session_state.google_authenticated:
            st.success("☁️ Cloud Ready")
        else:
            st.info("☁️ Cloud Offline")
    
    with download_col3:
        st.write("**🎯 Quick Actions:**")
        
        # Individual piece downloads
        if st.button("📱 Download First Piece"):
            first_piece_bytes = io.BytesIO()
            final_pieces[0].save(first_piece_bytes, format="PNG" if output_format == "PNG" else "JPEG", 
                          quality=quality_setting if output_format != "PNG" else None)
            st.download_button(
                "⬇️ Get First Piece",
                data=first_piece_bytes.getvalue(),
                file_name=f"IG_01.{'png' if output_format == 'PNG' else 'jpg'}",
                mime=f"image/{'png' if output_format == 'PNG' else 'jpeg'}"
            )
        
        # Template download with layer info
        if st.button("📋 Download Advanced Template"):
            template_content = f"""
INSTAGRAM GRID ADVANCED TEMPLATE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Grid Configuration:
- Layout: {rows} × {cols} ({len(final_pieces)} pieces)
- Output Size: {output_size}px
- Quality: {quality_setting}%
- Format: {output_format}

Advanced Features:
- Layers Applied: {len(st.session_state.layers)}
- Magic Eraser: {'Enabled' if st.session_state.magic_eraser_mode else 'Disabled'}
- Grid-Specific Layers: {'Yes' if any(layer['grid_pieces'] for layer in st.session_state.layers) else 'No'}

Enhancement Settings:
- Brightness: {enhance_brightness}
- Contrast: {enhance_contrast}  
- Saturation: {enhance_saturation}
- Sharpness: {enhance_sharpness}

Layer Details:
{chr(10).join([f"- {layer['type'].title()} Layer {layer['id']}: {layer['data'].get('text', 'N/A')[:20]}..." for layer in st.session_state.layers])}

Upload this template with your next project to maintain consistency.
            """
            st.download_button(
                "⬇️ Get Advanced Template",
                data=template_content,
                file_name=f"grid_advanced_template_{rows}x{cols}.txt",
                mime="text/plain"
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Enhanced Individual Downloads ---
    with st.expander("📁 Individual Piece Downloads & Advanced Cloud Sync"):
        individual_cols = st.columns(min(4, len(final_pieces)))
        
        for i, piece in enumerate(final_pieces):
            with individual_cols[i % len(individual_cols)]:
                # Add numbers if enabled
                if add_numbers:
                    display_piece = add_number_overlay(piece, i + 1, number_style, number_position)
                else:
                    display_piece = piece
                
                st.image(display_piece, use_container_width=True, caption=f"Part {i+1}")
                
                # Show which layers are applied to this piece
                applied_layers = [layer for layer in st.session_state.layers if not layer['grid_pieces'] or i in layer['grid_pieces']]
                if applied_layers:
                    st.caption(f"Layers: {len(applied_layers)} applied")
                
                # Download button
                img_bytes = io.BytesIO()
                if output_format == "PNG":
                    piece.save(img_bytes, format="PNG", optimize=True)
                    filename = f"IG_{i+1:02d}.png"
                    mime_type = "image/png"
                else:
                    piece.save(img_bytes, format="JPEG", quality=quality_setting, optimize=True)
                    filename = f"IG_{i+1:02d}.jpg"
                    mime_type = "image/jpeg"
                
                st.download_button(
                    f"⬇️ Download",
                    data=img_bytes.getvalue(),
                    file_name=filename,
                    mime=mime_type,
                    key=f"download_piece_{i}",
                    use_container_width=True
                )
                
                # Individual cloud upload
                if st.session_state.google_authenticated and GOOGLE_APIS_AVAILABLE:
                    if st.button(f"☁️ Upload", key=f"upload_piece_{i}", use_container_width=True):
                        with st.spinner(f"Uploading piece {i+1}..."):
                            try:
                                file_id, web_link, success = upload_to_drive(
                                    st.session_state.drive_service,
                                    img_bytes.getvalue(),
                                    filename,
                                    st.session_state.get('selected_folder_id'),
                                    mime_type
                                )
                                
                                if success:
                                    st.success(f"✅ Uploaded!")
                                else:
                                    st.error("❌ Upload failed")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

else:
    # --- Welcome Screen ---
    st.markdown("""
    <div class="feature-card">
        <h3>🚀 Welcome to Instagram Grid Splitter Pro+ Advanced!</h3>
        <p>Create professional Instagram grids with advanced layer management and cloud integration:</p>
        <ul>
            <li><strong>🎨 Advanced Layer System:</strong> Text, logo, and image layers with full editing capabilities</li>
            <li><strong>📋 Layer Management:</strong> Move, resize, reorder, and edit layers after creation</li>
            <li><strong>🖼️ Multiple Image Support:</strong> Add multiple images as layers with cropping and resizing</li>
            <li><strong>🪄 Magic Eraser:</strong> Remove unwanted objects with AI-powered erasing</li>
            <li><strong>🎯 Grid-Specific Layers:</strong> Apply layers to specific grid pieces only</li>
            <li><strong>☁️ Google Cloud Integration:</strong> Direct upload to Google Drive and project tracking</li>
            <li><strong>🌈 Professional Color Grading:</strong> Temperature, tint, highlights, and shadows</li>
            <li><strong>🎭 Advanced Filters:</strong> Vintage effects, vignettes, and Gaussian blur</li>
            <li><strong>📊 Analytics & Tracking:</strong> Quality analysis and project management</li>
            <li><strong>📱 Mobile Optimized:</strong> Perfect for Instagram's compression and mobile viewing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    showcase_col1, showcase_col2, showcase_col3 = st.columns(3)
    
    with showcase_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📋 Advanced Layer Management</h4>
            <p>Full layer editing with position controls, z-order management, text editing, and grid-specific placement.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with showcase_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🖼️ Multi-Image Support</h4>
            <p>Add multiple images as layers with cropping, resizing, and professional blending modes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with showcase_col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🪄 Magic Tools</h4>
            <p>Magic eraser for object removal and advanced image editing capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced features showcase
    advanced_col1, advanced_col2 = st.columns(2)
    
    with advanced_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Grid-Specific Editing</h4>
            <p>Apply different layers to specific grid pieces for ultimate customization and professional results.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with advanced_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>☁️ Cloud Integration</h4>
            <p>Seamless Google Drive uploads and Google Sheets project tracking with service account authentication.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Google Services Setup Guide
    if not GOOGLE_APIS_AVAILABLE:
        st.markdown("""
        <div class="warning-banner">
            <h4>📦 Google Services Setup Required</h4>
            <p>To enable Google Drive and Sheets integration, install the required packages:</p>
            <code>pip install google-api-python-client google-auth gspread</code>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample grid layouts
    st.subheader("📋 Available Grid Layouts")
    
    layout_examples = {
        "1×2": "Perfect for before/after, comparison posts, or story splits",
        "1×3": "Classic panoramic views, timelines, or step-by-step tutorials", 
        "2×2": "Balanced square layout for product showcases or mood boards",
        "2×3": "Rectangle grid ideal for storytelling or detailed product views",
        "3×3": "The ultimate Instagram grid for maximum impact and engagement",
        "1×4": "Extended horizontal strips for panoramic landscapes",
        "4×4": "Mega grid for complex artworks or detailed infographics"
    }
    
    for layout, description in layout_examples.items():
        st.markdown(f"**{layout} Grid:** {description}")
    
    st.warning("⬆️ **Upload an image to get started!** Supported formats: JPG, PNG, WebP, BMP")

# --- Footer Information ---
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    **📖 How to Use Advanced Features:**
    1. Upload main image and additional images
    2. Configure Google Cloud services (optional)
    3. Select grid pieces for layer targeting
    4. Add and edit text, logo, and image layers
    5. Use position controls and layer management
    6. Apply magic eraser and advanced filters
    7. Preview and download your professional grid!
    """)

with footer_col2:
    st.markdown("""
    **💡 Advanced Best Practices:**
    - Use layer management for professional results
    - Target specific grid pieces for unique effects
    - Edit layers after creation for perfect positioning
    - Use magic eraser to remove unwanted elements
    - Combine multiple images for complex compositions
    - Test different layer orders and blending modes
    """)

with footer_col3:
    st.markdown("""
    **🎯 Pro Advanced Features:**
    - Full layer editing and management system
    - Grid-specific layer placement
    - Multiple image layer support
    - Magic eraser and advanced tools
    - Real-time position controls
    - Professional layer ordering
    - Advanced cloud integration
    """)

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 8px;">
    <h4>🎨 Instagram Grid Splitter Pro+ Advanced</h4>
    <p><small>Professional grid creation with advanced layer management, multi-image support, and magic tools.</small></p>
    <p><strong>🔥 Advanced Features:</strong><br>
    ✅ Advanced Layer Management<br>
    ✅ Multi-Image Layer Support<br>
    ✅ Magic Eraser Tools<br>
    ✅ Grid-Specific Editing<br>
    ✅ Real-time Layer Controls<br>
    ✅ Professional Cloud Integration</p>
</div>
""", unsafe_allow_html=True)

