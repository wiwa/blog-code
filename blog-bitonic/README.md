Code for my [blog post](https://winwang.blog/posts/bitonic-sort) on CUDA bitonic sorting.

I assume C++ tools and CUDA are already installed.  
Feel free to open an issue if it doesn't work on your machine.

Build and run:

```sh
mkdir build
cd build
cmake ../cmake
cmake --build .
./BenchSort 24 8
```

The args mean 2^**24** elements, grid size of **8** blocks (32 threads). 
These elements are sorted in 32-element windows (non-overlapping)

This is not meant to be an example of a "production" benchmark (a la `criterion.rs`). 
Just something thrown together with fairly consistent results.
