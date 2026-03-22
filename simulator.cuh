#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <stdio.h>
#include <time.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;
typedef float f32;
typedef double f64;

struct block {u8 type; u32 id; u8 input_count; u32 inputs[255];};
struct blocks {
    u32 *states;  u32 *new_states;  u32 blocks_count;
    struct block *blocks;  u32 blocks_count_nor;
};

static __global__ void update_blocks(const struct block *__restrict__ blocks, u32 *__restrict__ states, u32 *__restrict__ new_states, u32 block_count, u32 steps) {
    u32 thread_id = (threadIdx.x+(blockIdx.x*blockDim.x));
    if (thread_id>=block_count) {return;}
    const block b = blocks[thread_id];
    u32 block_id = b.id;
    u8 state;
    u32 *tmp;
    for (;steps>0;steps--) {
        state=0;
        switch (b.input_count) {
            case 0:break;
            case 1:state=state|((states[b.inputs[0]>>5]>>(b.inputs[0]&31))&1);break;
            case 2:state=state|((states[b.inputs[0]>>5]>>(b.inputs[0]&31))&1)|((states[b.inputs[1]>>5]>>(b.inputs[1]&31))&1);break;
            case 3:state=state|((states[b.inputs[0]>>5]>>(b.inputs[0]&31))&1)|((states[b.inputs[1]>>5]>>(b.inputs[1]&31))&1)|((states[b.inputs[2]>>5]>>(b.inputs[2]&31))&1);break;
            case 4:state=state|((states[b.inputs[0]>>5]>>(b.inputs[0]&31))&1)|((states[b.inputs[1]>>5]>>(b.inputs[1]&31))&1)|((states[b.inputs[2]>>5]>>(b.inputs[2]&31))&1)|((states[b.inputs[3]>>5]>>(b.inputs[3]&31))&1);break;
            default:
                for (u32 i=0; i<b.input_count;) {
                    switch (b.input_count-i) {
                        case 1:state=state|((states[b.inputs[i]>>5]>>(b.inputs[i]&31))&1);i+=1;break;
                        case 2:state=state|((states[b.inputs[i]>>5]>>(b.inputs[i]&31))&1)|((states[b.inputs[i+1]>>5]>>(b.inputs[i+1]&31))&1);i+=2;break;
                        case 3:state=state|((states[b.inputs[i]>>5]>>(b.inputs[i]&31))&1)|((states[b.inputs[i+1]>>5]>>(b.inputs[i+1]&31))&1)|((states[b.inputs[i+2]>>5]>>(b.inputs[i+2]&31))&1);i+=3;break;
                        default:state=state|((states[b.inputs[i]>>5]>>(b.inputs[i]&31))&1)|((states[b.inputs[i+1]>>5]>>(b.inputs[i+1]&31))&1)|((states[b.inputs[i+2]>>5]>>(b.inputs[i+2]&31))&1)|((states[b.inputs[i+3]>>5]>>(b.inputs[i+3]&31))&1);i+=4;break;
                    }
                    if (state) {break;}
                }
        }
        switch (b.type) {
            case 0:new_states[block_id]=!state;
            case 6:new_states[block_id]=state;
        }
        tmp=states;
        states=new_states;
        new_states=tmp;
        __syncthreads();
    }
}

static __host__ struct blocks move_blocks_to_gpu(const struct blocks *h_blocks) {
    struct blocks d_blocks;
    d_blocks.blocks_count     = h_blocks->blocks_count;
    d_blocks.blocks_count_nor = h_blocks->blocks_count_nor;
    cudaMalloc(&d_blocks.states,((d_blocks.blocks_count*sizeof(u32))>>5)+1);
    cudaMalloc(&d_blocks.new_states,((d_blocks.blocks_count*sizeof(u32))>>5)+1);
    cudaMalloc(&d_blocks.blocks,d_blocks.blocks_count*sizeof(struct block));
    cudaMemcpy(d_blocks.states,h_blocks->states,((d_blocks.blocks_count*sizeof(u32))>>5)+1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_blocks.new_states,h_blocks->new_states,((d_blocks.blocks_count*sizeof(u32))>>5)+1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_blocks.blocks,h_blocks->blocks,d_blocks.blocks_count * sizeof(struct block),cudaMemcpyHostToDevice);
    return d_blocks;
}

static __host__ void free_blocks_on_gpu(struct blocks *d_blocks) {
    cudaFree(d_blocks->states);
    cudaFree(d_blocks->new_states);
    cudaFree(d_blocks->blocks);
    d_blocks->states = NULL;
    d_blocks->new_states = NULL;
    d_blocks->blocks = NULL;
}

static __host__ u32 ceil_u32(u32 x, u32 amount) {
    return (u32)(ceil((f64)x/(f64)amount)*(f64)amount);
}

u8 quit = 0;
struct blocks *device_blocks;
cudaStream_t stream;

static __host__ void simulator(cudaStream_t block_stream, blocks d_blocks, cudaStream_t state_stream) {
    for (u32 i = 0; i<1;) {
        if (quit) {break;}
        if (cudaStreamQuery(block_stream)==cudaSuccess) {
            update_blocks<<<(d_blocks.blocks_count+255)/256,256,0,block_stream>>>(d_blocks.blocks, d_blocks.states, d_blocks.new_states,d_blocks.blocks_count, 10000000);
            i++;
        }
    }
    for (;cudaStreamQuery(block_stream)!=cudaSuccess;) {}
    cudaStreamSynchronize(state_stream);
    free_blocks_on_gpu(&d_blocks);
}

__host__ void stop_simulator() {quit=1;}

__host__ void init_simulator(char *filepath) {
    FILE *ptr;
    ptr = fopen(filepath,"rb");
    struct blocks h_blocks;

    u32 block_count;
    u32 block_count_nor;
    u32 block_count_led;

    u8 read_buf[4];
    
    fread(read_buf, sizeof(u8), 4, ptr);
    block_count=(read_buf[0]<<24|read_buf[1]<<16|read_buf[2]<<8|read_buf[3]);
    fread(read_buf, sizeof(u8), 4, ptr);
    block_count_nor=(read_buf[0]<<24|read_buf[1]<<16|read_buf[2]<<8|read_buf[3]);
    fread(read_buf, sizeof(u8), 4, ptr);
    block_count_led=(read_buf[0]<<24|read_buf[1]<<16|read_buf[2]<<8|read_buf[3]);

    h_blocks.blocks_count=block_count;
    h_blocks.blocks_count_nor=block_count_nor;
    h_blocks.states = (u32*)malloc((ceil_u32(block_count*sizeof(u32),32)>>5));
    h_blocks.new_states = (u32*)malloc((ceil_u32(block_count*sizeof(u32),32)>>5));
    h_blocks.blocks = (struct block*)malloc((block_count) * sizeof(struct block));
    for (u32 i = 0; i < block_count; i++) {
        struct block b;
        if (i<block_count_nor) {b.type=0;}
        else if (i<block_count_nor+block_count_led) {b.type=6;}
        fread(read_buf, sizeof(u8), 4, ptr);
        b.id=(read_buf[0]<<24|read_buf[1]<<16|read_buf[2]<<8|read_buf[3]);
        fread(&b.input_count, sizeof(u8), 1, ptr);
        for (u8 j = 0; j < b.input_count; j++) {
            fread(read_buf, sizeof(u8), 4, ptr);
            b.inputs[j]=(read_buf[0]<<24|read_buf[1]<<16|read_buf[2]<<8|read_buf[3]);
        }
        fseek(ptr,12,SEEK_CUR);
        if (b.type==6) {fseek(ptr,5,SEEK_CUR);}
        h_blocks.blocks[i] = b;
    }
    fread(h_blocks.states, sizeof(u32), (ceil_u32(block_count*sizeof(u32),32)>>5), ptr);
    fread(h_blocks.new_states, sizeof(u32), (ceil_u32(block_count*sizeof(u32),32)>>5), ptr);
    fclose(ptr);
    struct blocks d_blocks = move_blocks_to_gpu(&h_blocks);
    device_blocks = &d_blocks;
    cudaStream_t state_stream;
    cudaStream_t block_stream;
    cudaStreamCreate(&state_stream);
    cudaStreamCreate(&block_stream);
    simulator(block_stream, d_blocks, state_stream);
}

__host__ void copy_states_to_cpu(u32 *host_states) {
    cudaMemcpyAsync(host_states,device_blocks->states,device_blocks->blocks_count/32+((device_blocks->blocks_count&31)>0),cudaMemcpyDeviceToHost,stream);
}