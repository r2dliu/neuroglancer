The following should already be included as the Shader

```
#uicontrol invlerp normalized(clamp=true)
void main() {
  float v = normalized(getDataValue(0));
  vec4 rgba = vec4(lut[int(v*255.)].r, lut[int(v*255.)].g, lut[int(v*255.)].b, 1.0);
  emitRGBA(rgba);
}
```

when loading this URL.

http://localhost:8080/#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B3156.4873046875%2C3793.629638671875%2C3543.705810546875%5D%2C%22crossSectionScale%22:3.762185354999915%2C%22projectionOrientation%22:%5B0.31435418128967285%2C0.8142172694206238%2C0.4843378961086273%2C-0.06040274351835251%5D%2C%22projectionScale%22:4593.980956070107%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/flyem_fib-25/image%22%2C%22tab%22:%22rendering%22%2C%22shader%22:%22#uicontrol%20invlerp%20normalized%28clamp=true%29%5Cnvoid%20main%28%29%20%7B%5Cn%20%20float%20v%20=%20normalized%28getDataValue%280%29%29%3B%5Cn%20%20vec4%20rgba%20=%20vec4%28lut%5Bint%28v%2A255.%29%5D.r%2C%20lut%5Bint%28v%2A255.%29%5D.g%2C%20lut%5Bint%28v%2A255.%29%5D.b%2C%201.0%29%3B%5Cn%20%20emitRGBA%28rgba%29%3B%5Cn%7D%22%2C%22name%22:%22image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/flyem_fib-25/ground_truth%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%2221894%22%2C%2222060%22%2C%22158571%22%2C%2224436%22%2C%222515%22%5D%2C%22name%22:%22ground-truth%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22size%22:390%2C%22visible%22:true%2C%22layer%22:%22image%22%7D%2C%22layout%22:%224panel%22%7D