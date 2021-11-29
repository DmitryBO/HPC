texture<unsigned int, 2> tex;

__global__ void kernel(unsigned int * __restrict__ image, const int M, const int N, const float sigma)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((x < M) && (y < N)) {


        float c = 0;
        float s = 0;

        for (int i = x-1; i <= x+1; i++){
            for (int j = y-1; j <= y+1; j++){
                
                float pxl1 = tex2D(tex, x, y);
                float pxl2 = tex2D(tex, j, i);

                float r = exp(-pow((pxl2 - pxl1), 2) / pow(sigma, 2));
                
                float g = exp(-(pow(j - x, 2) + pow(i - y, 2)) / pow(sigma, 2));
               
                c += g*r;
                s += g*r*tex2D(tex, j, i);
            }
        }
        image[x*N + y] = s / c;
    }
}