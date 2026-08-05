# L4 Evaluation Summary

## Performance Gates

| Gate | Status | Actual | Requirement |
|---|---|---:|---|
| bmm_long custom speedup | pass | 68.7198 | >= 10 |
| conv_resnet_block custom speedup | pass | 38.6291 | >= 10 |
| bmm_bert vendor/direct ratio | pass | 1.03619 | <= 5 |
| bmm_long vendor/direct ratio | pass | 1.06581 | <= 5 |
| bmm_value vendor/direct ratio | pass | 1.03854 | <= 5 |
| bmm_irregular vendor/direct ratio | pass | 1.03777 | <= 5 |
| conv_small vendor/direct ratio | pass | 0.987629 | <= 5 |
| conv_resnet_stem vendor/direct ratio | pass | 0.959782 | <= 5 |
| conv_resnet_block vendor/direct ratio | pass | 1.00798 | <= 5 |
| conv_pointwise vendor/direct ratio | pass | 1.01526 | <= 5 |
| conv_irregular vendor/direct ratio | pass | 1.03327 | <= 5 |
| PR2 GEMM p50 regression | pass | 0.135648 | 0.1206 to 0.1474 ms |

## Canonical Timings

| Family | Name | Backend | Source mode | Trials | p50 ms | p10 ms | p90 ms | GFLOP/s | p50 range |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| attention | attention_block | block-thread | block-thread | 3 | 1.948736 | 1.837280 | 1.992416 |  | 1.860704-1.964832 |
| attention | attention_block | pytorch-eager-cuda | pytorch | 3 | 0.087232 | 0.085952 | 0.100256 |  | 0.080448-0.095712 |
| attention | attention_block | untiled | untiled | 3 | 3.936512 | 3.907680 | 3.968416 |  | 3.841888-3.953536 |
| attention | attention_block | vendor | vendor | 3 | 1.897504 | 1.794208 | 1.957344 |  | 1.797952-1.898560 |
| bmm | bmm_bert | block-thread | block-thread | 3 | 0.043968 | 0.043072 | 0.044480 | 572.367 | 0.042944-0.044448 |
| bmm | bmm_bert | cublas-pedantic-fp32 | block-thread | 3 | 0.016576 | 0.016416 | 0.017056 | 1518.209 | 0.016544-0.016704 |
| bmm | bmm_bert | cublas-pedantic-fp32 | untiled | 3 | 0.016608 | 0.016416 | 0.016800 | 1515.283 | 0.016544-0.016608 |
| bmm | bmm_bert | cublas-pedantic-fp32 | vendor | 3 | 0.016800 | 0.016512 | 0.017664 | 1497.966 | 0.016704-0.016832 |
| bmm | bmm_bert | untiled | untiled | 3 | 1.036480 | 1.033856 | 1.039712 | 24.280 | 1.036480-1.037088 |
| bmm | bmm_bert | vendor | vendor | 3 | 0.017408 | 0.017216 | 0.017888 | 1445.647 | 0.017408-0.017504 |
| bmm | bmm_irregular | block-thread | block-thread | 3 | 0.036128 | 0.035360 | 0.037376 | 412.659 | 0.035840-0.036384 |
| bmm | bmm_irregular | cublas-pedantic-fp32 | block-thread | 3 | 0.017728 | 0.017600 | 0.018176 | 840.959 | 0.017600-0.017824 |
| bmm | bmm_irregular | cublas-pedantic-fp32 | untiled | 3 | 0.017664 | 0.017536 | 0.018112 | 844.006 | 0.017600-0.017760 |
| bmm | bmm_irregular | cublas-pedantic-fp32 | vendor | 3 | 0.017792 | 0.017664 | 0.018272 | 837.934 | 0.017728-0.018176 |
| bmm | bmm_irregular | untiled | untiled | 3 | 0.645856 | 0.644096 | 0.648032 | 23.083 | 0.645760-0.645888 |
| bmm | bmm_irregular | vendor | vendor | 3 | 0.018464 | 0.018272 | 0.018976 | 807.438 | 0.018432-0.019136 |
| bmm | bmm_long | block-thread | block-thread | 3 | 0.317472 | 0.316128 | 0.319008 | 1268.311 | 0.317152-0.323968 |
| bmm | bmm_long | cublas-pedantic-fp32 | block-thread | 3 | 0.042560 | 0.042016 | 0.043456 | 9460.836 | 0.042304-0.050112 |
| bmm | bmm_long | cublas-pedantic-fp32 | untiled | 3 | 0.042496 | 0.041920 | 0.043360 | 9475.084 | 0.042208-0.042784 |
| bmm | bmm_long | cublas-pedantic-fp32 | vendor | 3 | 0.042304 | 0.041920 | 0.042976 | 9518.087 | 0.042208-0.042752 |
| bmm | bmm_long | untiled | untiled | 3 | 21.816608 | 21.802176 | 21.828833 | 18.456 | 21.815680-21.816641 |
| bmm | bmm_long | vendor | vendor | 3 | 0.045088 | 0.044480 | 0.045856 | 8930.385 | 0.044928-0.045504 |
| bmm | bmm_value | block-thread | block-thread | 3 | 0.046624 | 0.046080 | 0.047552 | 719.682 | 0.046464-0.046688 |
| bmm | bmm_value | cublas-pedantic-fp32 | block-thread | 3 | 0.016512 | 0.016416 | 0.016896 | 2032.124 | 0.016512-0.016768 |
| bmm | bmm_value | cublas-pedantic-fp32 | untiled | 3 | 0.016544 | 0.016416 | 0.017312 | 2028.194 | 0.016544-0.016608 |
| bmm | bmm_value | cublas-pedantic-fp32 | vendor | 3 | 0.016608 | 0.016480 | 0.017152 | 2020.378 | 0.016576-0.016704 |
| bmm | bmm_value | untiled | untiled | 3 | 1.391904 | 1.388064 | 1.396128 | 24.107 | 1.391840-1.392640 |
| bmm | bmm_value | vendor | vendor | 3 | 0.017248 | 0.017024 | 0.017920 | 1945.410 | 0.017216-0.017280 |
| conv | conv_irregular | block-thread | block-thread | 3 | 0.026688 | 0.026432 | 0.027744 | 108.022 | 0.026528-0.026848 |
| conv | conv_irregular | cudnn-fp32 | block-thread | 3 | 0.032576 | 0.032224 | 0.033216 | 88.497 | 0.032480-0.034048 |
| conv | conv_irregular | cudnn-fp32 | untiled | 3 | 0.032288 | 0.032096 | 0.033056 | 89.286 | 0.032256-0.032416 |
| conv | conv_irregular | cudnn-fp32 | vendor | 3 | 0.032704 | 0.032320 | 0.033472 | 88.151 | 0.032512-0.032832 |
| conv | conv_irregular | untiled | untiled | 3 | 0.181056 | 0.180608 | 0.182208 | 15.923 | 0.181056-0.181248 |
| conv | conv_irregular | vendor | vendor | 3 | 0.033792 | 0.033408 | 0.034624 | 85.312 | 0.033664-0.034208 |
| conv | conv_pointwise | block-thread | block-thread | 3 | 0.079936 | 0.078592 | 0.081408 | 642.767 | 0.078720-0.080896 |
| conv | conv_pointwise | cudnn-fp32 | block-thread | 3 | 0.115680 | 0.114976 | 0.124608 | 444.158 | 0.115424-0.118368 |
| conv | conv_pointwise | cudnn-fp32 | untiled | 3 | 0.117568 | 0.116768 | 0.126912 | 437.026 | 0.116384-0.120128 |
| conv | conv_pointwise | cudnn-fp32 | vendor | 3 | 0.115328 | 0.114560 | 0.121760 | 445.514 | 0.115136-0.118336 |
| conv | conv_pointwise | untiled | untiled | 3 | 1.899168 | 1.776640 | 1.942592 | 27.054 | 1.896896-1.908480 |
| conv | conv_pointwise | vendor | vendor | 3 | 0.117088 | 0.116224 | 0.124448 | 438.817 | 0.116832-0.117120 |
| conv | conv_resnet_block | block-thread | block-thread | 3 | 0.278656 | 0.277152 | 0.281664 | 829.736 | 0.278560-0.279264 |
| conv | conv_resnet_block | cudnn-fp32 | block-thread | 3 | 0.050656 | 0.049952 | 0.051296 | 4564.336 | 0.050304-0.050688 |
| conv | conv_resnet_block | cudnn-fp32 | untiled | 3 | 0.051424 | 0.050784 | 0.052960 | 4496.169 | 0.051104-0.051488 |
| conv | conv_resnet_block | cudnn-fp32 | vendor | 3 | 0.052128 | 0.051456 | 0.053376 | 4435.448 | 0.050432-0.052256 |
| conv | conv_resnet_block | untiled | untiled | 3 | 10.764224 | 10.627072 | 10.915872 | 21.480 | 10.735648-10.768864 |
| conv | conv_resnet_block | vendor | vendor | 3 | 0.052544 | 0.051520 | 0.054752 | 4400.331 | 0.051200-0.052672 |
| conv | conv_resnet_stem | block-thread | block-thread | 3 | 0.226976 | 0.225664 | 0.231584 | 1039.880 | 0.226912-0.227136 |
| conv | conv_resnet_stem | cudnn-fp32 | block-thread | 3 | 0.064160 | 0.062944 | 0.065376 | 3678.739 | 0.063744-0.074016 |
| conv | conv_resnet_stem | cudnn-fp32 | untiled | 3 | 0.070016 | 0.069056 | 0.070848 | 3371.057 | 0.069952-0.070080 |
| conv | conv_resnet_stem | cudnn-fp32 | vendor | 3 | 0.064448 | 0.063040 | 0.065856 | 3662.300 | 0.063904-0.064608 |
| conv | conv_resnet_stem | untiled | untiled | 3 | 7.789248 | 7.778112 | 7.819072 | 30.302 | 7.750496-7.839072 |
| conv | conv_resnet_stem | vendor | vendor | 3 | 0.061856 | 0.060384 | 0.064288 | 3815.764 | 0.061504-0.061984 |
| conv | conv_small | block-thread | block-thread | 3 | 0.020160 | 0.019712 | 0.020736 | 89.600 | 0.019584-0.020160 |
| conv | conv_small | cudnn-fp32 | block-thread | 3 | 0.030208 | 0.029920 | 0.030976 | 59.797 | 0.030144-0.030336 |
| conv | conv_small | cudnn-fp32 | untiled | 3 | 0.030304 | 0.030048 | 0.031136 | 59.607 | 0.030080-0.030464 |
| conv | conv_small | cudnn-fp32 | vendor | 3 | 0.031040 | 0.029984 | 0.032224 | 58.194 | 0.030048-0.031136 |
| conv | conv_small | untiled | untiled | 3 | 0.087872 | 0.087200 | 0.088512 | 20.556 | 0.087424-0.087936 |
| conv | conv_small | vendor | vendor | 3 | 0.030656 | 0.030304 | 0.032672 | 58.923 | 0.030560-0.031648 |
| gemm | gemm_512 | block-thread | block-thread | 3 | 0.135648 | 0.134720 | 0.136960 | 989.456 | 0.135264-0.136128 |
| gemm | gemm_512 | cublas-pedantic-fp32 | block-thread | 3 | 0.027712 | 0.027328 | 0.028288 | 4843.307 | 0.027616-0.027968 |
| residual | residual_conv_block | block-thread | block-thread | 3 | 2.701440 | 1.876128 | 3.116768 |  | 2.407360-2.762848 |
| residual | residual_conv_block | pytorch-eager-cuda | pytorch | 3 | 0.182592 | 0.180736 | 0.194080 |  | 0.177088-0.190176 |
| residual | residual_conv_block | untiled | untiled | 3 | 3.129024 | 2.215936 | 3.697280 |  | 3.034144-3.466720 |
| residual | residual_conv_block | vendor | vendor | 3 | 2.293120 | 1.628320 | 3.528096 |  | 2.230816-2.600576 |
