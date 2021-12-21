# SB regular track Assignment 2 - Detection

Converting annotations to YOLOv5 format was done by a colleague (v3 annotations are in pixels, v5 annotations are in img ratios). Code/pseudocode for processing a single image:

```
x_center = (x+w/2)/image_w
y_center = (y+h/2)/image_h
width = w/image_w
height = h/image_h
```
