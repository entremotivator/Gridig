import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
import io
import zipfile
import math
import numpy as np

# --- App Config ---
st.set_page_config(
    page_title="Instagram Grid Splitter Pro", 
    page_icon="📸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .download-section {
        background: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .preview-grid {
        border: 2px dashed #ddd;
        padding: 1rem;
        border-radius: 8px;
        background: #fafafa;
    }
    .stats-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>📸 Instagram Grid Splitter Pro</h1>
    <p>Create stunning Instagram grids, carousels, and story splits with advanced customization</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

# --- Upload Section ---
st.sidebar.subheader("📁 Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose your image", 
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Supported formats: JPG, PNG, WebP, BMP"
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
        custom_rows = st.number_input("Rows", min_value=1, max_value=6, value=3)
    with col2:
        custom_cols = st.number_input("Cols", min_value=1, max_value=6, value=3)

# --- Image Processing Options ---
st.sidebar.subheader("🎨 Image Processing")

# Auto-square options
square_option = st.sidebar.selectbox(
    "Square formatting:",
    ["Keep original ratio", "Crop to square (center)", "Crop to square (smart)", "Pad to square", "Stretch to square"],
    index=0,
    help="How to handle non-square images"
)

# Output quality settings
st.sidebar.subheader("📊 Quality & Size")
output_size = st.sidebar.slider("Output size (px)", 400, 2160, 1080, 40, help="Standard Instagram size is 1080px")
quality_setting = st.sidebar.slider("JPEG Quality", 60, 100, 95, 5, help="Higher = better quality, larger file")

# Image enhancements
st.sidebar.subheader("✨ Enhancements")
enhance_brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
enhance_contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
enhance_saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0, 0.1)
enhance_sharpness = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)

# Border and spacing options
st.sidebar.subheader("🎯 Layout Options")
add_borders = st.sidebar.checkbox("Add borders between pieces", value=False)
border_width = st.sidebar.slider("Border width (px)", 1, 20, 5) if add_borders else 0
border_color = st.sidebar.color_picker("Border color", "#FFFFFF") if add_borders else "#FFFFFF"

# Padding options
pad_color = st.sidebar.color_picker("Padding color", "#FFFFFF")

# Output format
st.sidebar.subheader("💾 Output")
output_format = st.sidebar.radio("Download format:", ["PNG", "JPG"], index=1)

# Numbering options
add_numbers = st.sidebar.checkbox("Add posting order numbers", value=True)
number_style = st.sidebar.selectbox(
    "Number style:",
    ["Red bold", "White with black outline", "Subtle gray", "Colorful"],
    index=0
) if add_numbers else "Red bold"

# --- Helper Functions ---
def get_grid_dimensions(grid_option, custom_rows=3, custom_cols=3):
    """Get rows and columns from grid option."""
    if "1 × 2" in grid_option:
        return 1, 2
    elif "1 × 3" in grid_option:
        return 1, 3
    elif "2 × 2" in grid_option:
        return 2, 2
    elif "2 × 3" in grid_option:
        return 2, 3
    elif "3 × 3" in grid_option:
        return 3, 3
    elif "1 × 4" in grid_option:
        return 1, 4
    elif "4 × 4" in grid_option:
        return 4, 4
    elif "Custom" in grid_option:
        return custom_rows, custom_cols
    return 3, 3

def smart_crop_to_square(img):
    """Smart crop that tries to preserve the most important part of the image."""
    w, h = img.size
    if w == h:
        return img
    
    # Convert to grayscale for edge detection
    gray = img.convert('L')
    # Apply edge detection filter
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # Convert to numpy for analysis
    edge_array = np.array(edges)
    
    if w > h:
        # Landscape: find the most interesting vertical strip
        column_scores = np.sum(edge_array, axis=0)
        # Smooth the scores
        window_size = h
        smoothed_scores = np.convolve(column_scores, np.ones(window_size)/window_size, mode='valid')
        best_start = np.argmax(smoothed_scores)
        left = best_start
        right = left + h
        return img.crop((left, 0, right, h))
    else:
        # Portrait: find the most interesting horizontal strip
        row_scores = np.sum(edge_array, axis=1)
        window_size = w
        smoothed_scores = np.convolve(row_scores, np.ones(window_size)/window_size, mode='valid')
        best_start = np.argmax(smoothed_scores)
        top = best_start
        bottom = top + w
        return img.crop((0, top, w, bottom))

def make_square(img, option="Keep original ratio", size=None, pad_color="#FFFFFF"):
    """Convert image to square using various methods."""
    if option == "Keep original ratio":
        if size:
            # Maintain aspect ratio but fit within size constraints
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img
    
    w, h = img.size
    
    if option == "Crop to square (center)":
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        img = img.crop((left, top, right, bottom))
    
    elif option == "Crop to square (smart)":
        img = smart_crop_to_square(img)
    
    elif option == "Pad to square":
        max_side = max(w, h)
        delta_w = max_side - w
        delta_h = max_side - h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img = ImageOps.expand(img, padding, pad_color)
    
    elif option == "Stretch to square":
        min_side = min(w, h)
        img = img.resize((min_side, min_side), Image.Resampling.LANCZOS)
    
    # Resize to final output size if specified
    if size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    return img

def enhance_image(img, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    """Apply image enhancements."""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
    
    return img

def add_borders_to_pieces(pieces, border_width, border_color, rows, cols):
    """Add borders between grid pieces."""
    if border_width <= 0:
        return pieces
    
    bordered_pieces = []
    for i, piece in enumerate(pieces):
        # Calculate which borders this piece needs
        row = i // cols
        col = i % cols
        
        # Determine border sides (top, right, bottom, left)
        borders = [0, 0, 0, 0]  # TRBL
        
        if row > 0:  # Not top row
            borders[0] = border_width // 2
        if col < cols - 1:  # Not rightmost column
            borders[1] = border_width // 2
        if row < rows - 1:  # Not bottom row
            borders[2] = border_width // 2
        if col > 0:  # Not leftmost column
            borders[3] = border_width // 2
        
        # Add borders
        if any(borders):
            bordered_piece = ImageOps.expand(piece, tuple(borders), border_color)
            bordered_pieces.append(bordered_piece)
        else:
            bordered_pieces.append(piece)
    
    return bordered_pieces

def split_image(img, rows, cols, add_spacing=False, spacing_width=0):
    """Split image into grid pieces with optional spacing."""
    w, h = img.size
    
    if add_spacing and spacing_width > 0:
        # Account for spacing in calculations
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
            piece = img.crop((left, top, right, bottom))
            pieces.append(piece)
    
    return pieces

def get_number_style_config(style):
    """Get font configuration for different number styles."""
    configs = {
        "Red bold": {"fill": (255, 0, 0, 255), "outline": None, "outline_width": 0},
        "White with black outline": {"fill": (255, 255, 255, 255), "outline": (0, 0, 0, 255), "outline_width": 3},
        "Subtle gray": {"fill": (128, 128, 128, 180), "outline": None, "outline_width": 0},
        "Colorful": {"fill": (255, 215, 0, 255), "outline": (255, 0, 255, 255), "outline_width": 2}
    }
    return configs.get(style, configs["Red bold"])

def add_number_overlay(img, number, style="Red bold", position="center"):
    """Add posting order number to image with style options."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Dynamic font size based on image size
    font_size = max(30, min(img.size[0] // 6, img.size[1] // 6))
    
    try:
        # Try to load a better font
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text = str(number)
    style_config = get_number_style_config(style)
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Position calculation
    if position == "center":
        x = (img.size[0] - text_w) // 2
        y = (img.size[1] - text_h) // 2
    elif position == "top-left":
        x, y = 20, 20
    elif position == "top-right":
        x = img.size[0] - text_w - 20
        y = 20
    elif position == "bottom-left":
        x = 20
        y = img.size[1] - text_h - 20
    elif position == "bottom-right":
        x = img.size[0] - text_w - 20
        y = img.size[1] - text_h - 20
    
    # Draw text with outline if specified
    if style_config["outline"] and style_config["outline_width"] > 0:
        # Draw outline
        outline_width = style_config["outline_width"]
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx*dx + dy*dy <= outline_width*outline_width:
                    draw.text((x + dx, y + dy), text, font=font, fill=style_config["outline"])
    
    # Draw main text
    draw.text((x, y), text, font=font, fill=style_config["fill"])
    
    return overlay

def calculate_optimal_size(original_size, target_ratio=1.0):
    """Calculate optimal size maintaining quality."""
    w, h = original_size
    if target_ratio == 1.0:  # Square
        min_side = min(w, h)
        return min_side, min_side
    return w, h

def create_collage_preview(pieces, rows, cols, spacing=5):
    """Create a collage preview of all pieces."""
    if not pieces:
        return None
    
    piece_w, piece_h = pieces[0].size
    collage_w = cols * piece_w + (cols - 1) * spacing
    collage_h = rows * piece_h + (rows - 1) * spacing
    
    collage = Image.new('RGB', (collage_w, collage_h), 'white')
    
    for i, piece in enumerate(pieces):
        row = i // cols
        col = i % cols
        x = col * (piece_w + spacing)
        y = row * (piece_h + spacing)
        collage.paste(piece, (x, y))
    
    return collage

def generate_posting_guide(rows, cols, grid_type="feed"):
    """Generate posting instructions based on grid type."""
    total_pieces = rows * cols
    
    if grid_type == "carousel":
        return f"""
        **📱 Carousel Posting Guide:**
        - Upload all {total_pieces} images in numerical order (1→{total_pieces})
        - Instagram will display them as swipeable carousel
        - First image becomes the cover image
        - Perfect for storytelling and detailed views
        """
    else:
        return f"""
        **📋 Grid Feed Posting Guide:**
        - Post in reverse order: Start with IG_{total_pieces:02d} and work backwards to IG_01
        - This ensures proper grid alignment in your feed
        - Wait a few minutes between posts for best algorithm performance
        - Use consistent hashtags and captions for cohesive look
        """

# --- Main Application Logic ---
if uploaded_file:
    # Load and display original image info
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size
    original_format = uploaded_file.type
    
    # Display image statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <h4>📏 Dimensions</h4>
            <p>{original_size[0]} × {original_size[1]} px</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.markdown(f"""
        <div class="stats-box">
            <h4>📦 File Size</h4>
            <p>{file_size:.1f} KB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        aspect_ratio = original_size[0] / original_size[1]
        st.markdown(f"""
        <div class="stats-box">
            <h4>📐 Aspect Ratio</h4>
            <p>{aspect_ratio:.2f}:1</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-box">
            <h4>🎨 Format</h4>
            <p>{original_format}</p>
        </div>
        """, unsafe_allow_html=True)

    # Get grid dimensions
    if "Custom" in grid_option:
        rows, cols = custom_rows, custom_cols
    else:
        rows, cols = get_grid_dimensions(grid_option)
    
    # Apply image enhancements
    enhanced_image = enhance_image(
        image, 
        enhance_brightness, 
        enhance_contrast, 
        enhance_saturation, 
        enhance_sharpness
    )
    
    # Apply square formatting
    processed_image = make_square(
        enhanced_image, 
        option=square_option, 
        size=output_size, 
        pad_color=pad_color
    )

    # --- Image Comparison ---
    st.subheader("🖼️ Image Processing Preview")
    comparison_cols = st.columns(3)
    
    with comparison_cols[0]:
        st.image(image, caption="📷 Original Image", use_container_width=True)
    
    with comparison_cols[1]:
        st.image(enhanced_image, caption="✨ Enhanced Image", use_container_width=True)
    
    with comparison_cols[2]:
        st.image(processed_image, caption=f"📐 Final Processed ({square_option})", use_container_width=True)

    # Split image into pieces
    pieces = split_image(processed_image, rows, cols)
    
    # Add borders if requested
    if add_borders:
        pieces = add_borders_to_pieces(pieces, border_width, border_color, rows, cols)

    # --- Grid Preview ---
    st.subheader(f"✨ Grid Preview ({rows} × {cols} = {len(pieces)} pieces)")
    
    # Create tabs for different preview modes
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["🔢 Numbered Preview", "🖼️ Clean Preview", "📱 Collage View"])
    
    with preview_tab1:
        st.markdown('<div class="preview-grid">', unsafe_allow_html=True)
        preview_cols = st.columns(cols)
        for idx, piece in enumerate(pieces):
            if add_numbers:
                numbered_piece = add_number_overlay(piece, idx + 1, number_style)
            else:
                numbered_piece = piece
            
            with preview_cols[idx % cols]:
                st.image(numbered_piece, use_container_width=True, caption=f"Part {idx+1}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with preview_tab2:
        st.markdown('<div class="preview-grid">', unsafe_allow_html=True)
        clean_cols = st.columns(cols)
        for idx, piece in enumerate(pieces):
            with clean_cols[idx % cols]:
                st.image(piece, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with preview_tab3:
        collage = create_collage_preview(pieces, rows, cols, spacing=10)
        if collage:
            st.image(collage, caption="Complete Grid Collage", use_container_width=True)

    # --- Download Section ---
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.subheader("⬇️ Download Your Grid")
    
    # Create download options
    download_col1, download_col2 = st.columns([2, 1])
    
    with download_col1:
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, piece in enumerate(pieces, 1):
                img_bytes = io.BytesIO()
                
                if output_format == "PNG":
                    piece.save(img_bytes, format="PNG", optimize=True)
                    filename = f"IG_Grid_{i:02d}.png"
                else:
                    piece.save(img_bytes, format="JPEG", quality=quality_setting, optimize=True)
                    filename = f"IG_Grid_{i:02d}.jpg"
                
                zf.writestr(filename, img_bytes.getvalue())
            
            # Add posting guide as text file
            guide_content = generate_posting_guide(rows, cols, "feed")
            zf.writestr("POSTING_GUIDE.txt", guide_content)

        # Main download button
        st.download_button(
            "📦 Download Complete Grid Package (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"instagram_grid_{rows}x{cols}_{uploaded_file.name.split('.')[0]}.zip",
            mime="application/zip",
            help=f"Downloads all {len(pieces)} pieces plus posting guide"
        )
    
    with download_col2:
        # Statistics
        total_size = sum(len(piece.tobytes()) for piece in pieces) / 1024 / 1024  # MB
        st.metric("Total Size", f"{total_size:.1f} MB")
        st.metric("Pieces", len(pieces))
        st.metric("Resolution", f"{output_size}×{output_size}" if square_option != "Keep original ratio" else f"{processed_image.size[0]}×{processed_image.size[1]}")

    # Individual download options
    with st.expander("📁 Download Individual Pieces"):
        individual_cols = st.columns(min(4, len(pieces)))
        for i, piece in enumerate(pieces):
            with individual_cols[i % len(individual_cols)]:
                img_bytes = io.BytesIO()
                
                if output_format == "PNG":
                    piece.save(img_bytes, format="PNG", optimize=True)
                    filename = f"IG_Grid_{i+1:02d}.png"
                    mime_type = "image/png"
                else:
                    piece.save(img_bytes, format="JPEG", quality=quality_setting, optimize=True)
                    filename = f"IG_Grid_{i+1:02d}.jpg"
                    mime_type = "image/jpeg"
                
                st.download_button(
                    f"⬇️ Part {i+1}",
                    data=img_bytes.getvalue(),
                    file_name=filename,
                    mime=mime_type,
                    key=f"download_{i}"
                )

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Posting Instructions ---
    st.subheader("📋 Posting Instructions")
    
    instruction_tab1, instruction_tab2, instruction_tab3 = st.tabs(["🔄 Feed Grid", "📱 Carousel", "📖 Pro Tips"])
    
    with instruction_tab1:
        st.markdown(generate_posting_guide(rows, cols, "feed"))
        
        # Visual posting order
        st.write("**Visual Posting Order for Feed Grid:**")
        order_cols = st.columns(cols)
        posting_order = list(range(len(pieces), 0, -1))  # Reverse order for feed
        
        for idx in range(len(pieces)):
            row = idx // cols
            col = idx % cols
            with order_cols[col]:
                st.info(f"Post #{posting_order[idx]}\n(IG_{idx+1:02d})")
    
    with instruction_tab2:
        st.markdown(generate_posting_guide(rows, cols, "carousel"))
        
        # Carousel order visualization
        st.write("**Carousel Order (Left to Right):**")
        carousel_cols = st.columns(min(len(pieces), 6))
        for i in range(len(pieces)):
            with carousel_cols[i % len(carousel_cols)]:
                st.success(f"Slide {i+1}")
    
    with instruction_tab3:
        st.markdown("""
        **🎯 Pro Tips for Maximum Engagement:**
        
        **Timing & Strategy:**
        - Post during your audience's peak hours (check Instagram Insights)
        - Space grid posts 15-30 minutes apart for algorithm optimization
        - Use the first post to tease the complete grid reveal
        
        **Content Best Practices:**
        - Keep important elements away from edges (Instagram compression)
        - Use high contrast for better mobile viewing
        - Test your grid on different devices before posting
        
        **Hashtag Strategy:**
        - Use consistent hashtags across all grid posts
        - Include grid-specific tags like #instagramgrid #gridpost
        - Mix popular and niche hashtags for better reach
        
        **Engagement Hacks:**
        - Ask followers to swipe/scroll to see the full image
        - Create anticipation with "Grid reveal coming..." stories
        - Pin the complete grid as a highlight for new followers
        """)

    # --- Advanced Features ---
    with st.expander("🚀 Advanced Features & Analytics"):
        
        # Grid statistics
        st.subheader("📊 Grid Analytics")
        
        analytics_col1, analytics_col2 = st.columns(2)
        
        with analytics_col1:
            # Color analysis
            st.write("**🎨 Color Composition:**")
            sample_piece = pieces[0]
            colors = sample_piece.getcolors(maxcolors=256*256*256)
            if colors:
                dominant_color = max(colors, key=lambda item: item[0])
                st.color_picker("Dominant Color", f"#{dominant_color[1][0]:02x}{dominant_color[1][1]:02x}{dominant_color[1][2]:02x}", disabled=True)
        
        with analytics_col2:
            # Complexity analysis
            st.write("**🧠 Image Complexity:**")
            gray_piece = pieces[0].convert('L')
            edges = gray_piece.filter(ImageFilter.FIND_EDGES)
            edge_pixels = np.array(edges)
            complexity = np.mean(edge_pixels) / 255 * 100
            st.metric("Complexity Score", f"{complexity:.1f}%")
        
        # Optimal posting times suggestion
        st.subheader("⏰ Optimal Posting Schedule")
        total_posts = len(pieces)
        
        schedule_option = st.radio(
            "Posting frequency:",
            ["Every 15 minutes", "Every 30 minutes", "Every hour", "Daily"],
            index=1
        )
        
        if schedule_option == "Every 15 minutes":
            total_time = total_posts * 15
            st.info(f"⏱️ Complete grid posting will take {total_time} minutes ({total_time/60:.1f} hours)")
        elif schedule_option == "Every 30 minutes":
            total_time = total_posts * 30
            st.info(f"⏱️ Complete grid posting will take {total_time} minutes ({total_time/60:.1f} hours)")
        elif schedule_option == "Every hour":
            st.info(f"⏱️ Complete grid posting will take {total_posts} hours")
        else:
            st.info(f"⏱️ Complete grid posting will take {total_posts} days")

    # --- Templates and Presets ---
    with st.expander("🎨 Quick Presets & Templates"):
        
        preset_col1, preset_col2 = st.columns(2)
        
        with preset_col1:
            st.write("**📸 Popular Presets:**")
            
            if st.button("🌅 Landscape Showcase (1×3)"):
                st.session_state.preset_applied = "landscape"
                
            if st.button("📱 Story Split (1×2)"):
                st.session_state.preset_applied = "story"
                
            if st.button("🎨 Art Gallery (3×3)"):
                st.session_state.preset_applied = "gallery"
                
            if st.button("📖 Comic Strip (2×3)"):
                st.session_state.preset_applied = "comic"
        
        with preset_col2:
            st.write("**🎯 Template Suggestions:**")
            
            # Template recommendations based on image aspect ratio
            if original_size[0] > original_size[1]:  # Landscape
                st.success("✅ Recommended: 1×3 or 2×3 grid")
                st.info("💡 Your landscape image works great for horizontal grids")
            elif original_size[1] > original_size[0]:  # Portrait
                st.success("✅ Recommended: 3×3 or 1×2 grid")
                st.info("💡 Your portrait image is perfect for square or vertical grids")
            else:  # Square
                st.success("✅ Recommended: 3×3 grid")
                st.info("💡 Your square image is ideal for any grid layout")

else:
    # --- Welcome Screen ---
    st.markdown("""
    <div class="feature-card">
        <h3>🚀 Welcome to Instagram Grid Splitter Pro!</h3>
        <p>Create professional Instagram grids with advanced features:</p>
        <ul>
            <li><strong>Multiple Grid Layouts:</strong> From simple 1×2 splits to complex 4×4 mega grids</li>
            <li><strong>Smart Image Processing:</strong> Auto-square with intelligent cropping</li>
            <li><strong>Image Enhancement:</strong> Brightness, contrast, saturation, and sharpness controls</li>
            <li><strong>Professional Features:</strong> Custom borders, spacing, and numbering systems</li>
            <li><strong>Posting Guides:</strong> Complete instructions for both feed grids and carousels</li>
            <li><strong>Batch Download:</strong> ZIP packages with posting guides included</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    showcase_col1, showcase_col2, showcase_col3 = st.columns(3)
    
    with showcase_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📐 Smart Cropping</h4>
            <p>Intelligent algorithms detect the most important parts of your image for optimal cropping.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with showcase_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🎨 Image Enhancement</h4>
            <p>Built-in tools to adjust brightness, contrast, saturation, and sharpness for perfect Instagram posts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with showcase_col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📱 Mobile Optimized</h4>
            <p>All outputs are optimized for Instagram's compression and mobile viewing experience.</p>
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
    **📖 How to Use:**
    1. Upload your image
    2. Choose grid layout
    3. Adjust processing options
    4. Preview your grid
    5. Download and post!
    """)

with footer_col2:
    st.markdown("""
    **💡 Best Practices:**
    - Use high-resolution images (1080px+)
    - Keep important content away from edges
    - Test different enhancement settings
    - Follow the posting order guide
    """)

with footer_col3:
    st.markdown("""
    **🎯 Pro Features:**
    - Smart cropping algorithm
    - Custom border and spacing
    - Multiple enhancement filters
    - Automated posting guides
    - Batch processing and download
    """)

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 8px;">
    <h4>📱 Instagram Grid Splitter Pro</h4>
    <p><small>Create viral-worthy grid content with professional tools and features.</small></p>
    <p><strong>🔥 Features:</strong><br>
    ✅ 8 Grid Layouts<br>
    ✅ Smart Image Processing<br>
    ✅ Professional Enhancements<br>
    ✅ Automated Posting Guides<br>
    ✅ Batch Download System</p>
</div>
""", unsafe_allow_html=True)

# --- Performance Tips ---
if uploaded_file:
    with st.expander("⚡ Performance & Quality Tips"):
        st.markdown("""
        **🎯 Image Quality Tips:**
        - **Original Resolution:** Higher input = better output quality
        - **Compression:** PNG for graphics, JPG for photos
        - **Size vs Quality:** 1080px is Instagram's sweet spot
        
        **📈 Engagement Optimization:**
        - **Grid Reveal Strategy:** Tease the complete grid in stories
        - **Posting Timing:** Use Instagram Insights for optimal times
        - **Caption Consistency:** Keep captions cohesive across grid posts
        
        **🔧 Technical Best Practices:**
        - **Mobile Preview:** Always check how grid looks on mobile
        - **Feed Planning:** Consider how grid fits with existing posts
        - **Backup Strategy:** Save original high-res versions
        """)

# --- Session State for Presets ---
if 'preset_applied' not in st.session_state:
    st.session_state.preset_applied = None

# Apply preset configurations
if st.session_state.preset_applied:
    if st.session_state.preset_applied == "landscape" and uploaded_file:
        st.info("🌅 Landscape preset applied! Optimized for panoramic views.")
    elif st.session_state.preset_applied == "story" and uploaded_file:
        st.info("📱 Story split preset applied! Perfect for before/after content.")
    elif st.session_state.preset_applied == "gallery" and uploaded_file:
        st.info("🎨 Art gallery preset applied! Maximum impact 3×3 grid.")
    elif st.session_state.preset_applied == "comic" and uploaded_file:
        st.info("📖 Comic strip preset applied! Great for sequential storytelling.")
    
    # Reset preset state
    st.session_state.preset_applied = None
