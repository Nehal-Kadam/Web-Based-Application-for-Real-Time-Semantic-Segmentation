import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# If the user does not provide both input and output image arguments, exit
if len(sys.argv) != 3:
    print("Usage: python3 segment.py <input_image> <output_image>")
    exit(1)

# Initialize the SamGeo model (provide the checkpoint path if needed)
from samgeo import SamGeo

# Assuming the checkpoint file is provided as a parameter
checkpoint_path = 'sam_vit_h_4b8939.pth'  # Provide the correct checkpoint path here
sam = SamGeo(checkpoint=checkpoint_path)

# Generate segmentation results
sam.generate(sys.argv[1])

# Save the segmentation masks
sam.save_masks()

# Define custom colors for specific terrain types
custom_colors = [
    (0.0, 1.0, 0.0),  # Green for trees/vegetation
    (0.5, 0.0, 0.5),  # Yellow for buildings
    (1.0, 1.0, 0.0),  # Yellow for streets (same as buildings for your request)
    (0.0, 0.0, 1.0),  # Blue for water bodies
    (1.0, 0.0, 0.0),  # Red for other roads (optional)
    
    (0.0, 1.0, 1.0),  # Cyan for forest areas
    
]

# Create a custom colormap with the defined colors
cmap = ListedColormap(custom_colors)

# Save the segmented image with the custom colormap
plt.imsave(sys.argv[2], sam.objects, cmap=cmap)

# Optional: Display the segmented image (useful for debugging or visualization)
plt.imshow(sam.objects, cmap=cmap)
plt.axis('off')  # Hide axes for cleaner display
plt.show()
