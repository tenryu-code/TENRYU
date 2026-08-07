#pragma once

#ifdef __CUDACC__
#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE __host__ __device__
#endif
#else
#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE
#endif
#endif

