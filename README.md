# FiftyOne-AI-Image-Duplicate-Checker
This is a Python script that utilizes the FiftyOne library to identify and remove duplicate images from a given dataset based on cosine similarity between image embeddings.

## Notes

    Ensure that your image folder contains only image files (PNG, JPG, JPEG, or WebP).
    The script calculates image embeddings using the CLIP-ViT model from the FiftyOne Zoo.
    Images with similarity above the specified threshold will be marked for deletion.
