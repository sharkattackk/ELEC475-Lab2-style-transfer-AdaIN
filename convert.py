from PIL import Image
img = Image.open('./images/content/mini.png')

# Convert the image to RGB (removing the alpha channel)
img = img.convert('RGB')

# Save the converted image
img.save('mini.png')