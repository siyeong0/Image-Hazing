# Image Hazing using depthmap and random noise


```python
# Function signiture
# Returns hazed cv image
def imHaze(path, intensity=3, CAMERA_ALTITUDE=3.5, HORIZONTAL_ANGLE=0, CAMERA_VERTICAL_FOV=64):

# Example
hazed_image = imHaze("test.jpg")
```

|Before|After|
|-----|-----|
|![test](https://user-images.githubusercontent.com/117014820/224487259-f20bae5c-bd44-4af4-88a1-af5de0917e97.jpg)|![result](https://user-images.githubusercontent.com/117014820/224487277-db58f25b-bbc8-495d-8f4c-4a30780839a1.jpg)|
| ![test](https://user-images.githubusercontent.com/117014820/224486735-cc0d7dff-9612-4d69-9b50-bb59d09ef9ea.jpg)| ![result](https://user-images.githubusercontent.com/117014820/224486796-c3daa258-6b8b-446a-8b60-512f594537b3.jpg)|
