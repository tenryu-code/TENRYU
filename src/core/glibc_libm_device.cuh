#pragma once

#include <math_constants.h>

// exp, log and pow on the device with the results of the x86-64 GNU C library (glibc 2.39, the
// build its ifunc selects on processors with FMA and AVX2): the same tables, the same operations
// in the same order, the same fused multiply-adds, so that a host computation that called the
// glibc functions gives the same bits when it moves to the device (CUDA's own exp, log and pow
// differ from glibc's by one unit in the last place for 6 % to 26 % of the arguments of exp and
// pow, 2026-10-01 measurement).
//
// The algorithms and tables are those of Arm's optimized-routines (release v19.11, math/exp.c,
// exp_data.c, log.c, log_data.c, pow.c, pow_log_data.c), which glibc 2.28 and later use; the
// 931 table values below were checked equal, bit for bit, to the data of glibc 2.39's libm.so.6.
// Where the C source leaves the compiler free to fuse a multiply and an add, the order of the
// operations and the fused multiply-adds follow the machine code of glibc 2.39's FMA build of
// exp, log and pow (Ubuntu 2.39-0ubuntu8, x86-64), read instruction by instruction. The special
// cases return the IEEE results of the C source with the x86 rules for NaN operands (a NaN operand
// is returned with its quiet bit set; an invalid operation on numbers gives the x86 default NaN
// 0xfff8000000000000). The status flags and errno of the library are not reproduced.
//
// Copyright (c) 2018, Arm Limited (the algorithms and the tables). SPDX-License-Identifier: MIT.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions: The above copyright notice
// and this permission notice shall be included in all copies or substantial portions of the
// Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
// AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
// OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

namespace tenryu::core::glibc_libm {

namespace detail {

// exp: N/ln2, the rounding shift, -ln2/N (high and low part), C2..C5, and the 2^(i/N) table
// (tab[2 i] the tail bits, tab[2 i + 1] the scale bits minus i << 45), N = 128.
static __device__ const unsigned long long kExpConst[8] = {
    0x40671547652b82feULL, 0x4338000000000000ULL, 0xbf762e42fefa0000ULL, 0xbd0cf79abc9e3b3aULL,
    0x3fdffffffffffdbdULL, 0x3fc555555555543cULL, 0x3fa55555cf172b91ULL, 0x3f81111167a4d017ULL};
static __device__ const unsigned long long kExpTab[256] = {
    0x0000000000000000ULL, 0x3ff0000000000000ULL, 0x3c9b3b4f1a88bf6eULL, 0x3feff63da9fb3335ULL,
    0xbc7160139cd8dc5dULL, 0x3fefec9a3e778061ULL, 0xbc905e7a108766d1ULL, 0x3fefe315e86e7f85ULL,
    0x3c8cd2523567f613ULL, 0x3fefd9b0d3158574ULL, 0xbc8bce8023f98efaULL, 0x3fefd06b29ddf6deULL,
    0x3c60f74e61e6c861ULL, 0x3fefc74518759bc8ULL, 0x3c90a3e45b33d399ULL, 0x3fefbe3ecac6f383ULL,
    0x3c979aa65d837b6dULL, 0x3fefb5586cf9890fULL, 0x3c8eb51a92fdeffcULL, 0x3fefac922b7247f7ULL,
    0x3c3ebe3d702f9cd1ULL, 0x3fefa3ec32d3d1a2ULL, 0xbc6a033489906e0bULL, 0x3fef9b66affed31bULL,
    0xbc9556522a2fbd0eULL, 0x3fef9301d0125b51ULL, 0xbc5080ef8c4eea55ULL, 0x3fef8abdc06c31ccULL,
    0xbc91c923b9d5f416ULL, 0x3fef829aaea92de0ULL, 0x3c80d3e3e95c55afULL, 0x3fef7a98c8a58e51ULL,
    0xbc801b15eaa59348ULL, 0x3fef72b83c7d517bULL, 0xbc8f1ff055de323dULL, 0x3fef6af9388c8deaULL,
    0x3c8b898c3f1353bfULL, 0x3fef635beb6fcb75ULL, 0xbc96d99c7611eb26ULL, 0x3fef5be084045cd4ULL,
    0x3c9aecf73e3a2f60ULL, 0x3fef54873168b9aaULL, 0xbc8fe782cb86389dULL, 0x3fef4d5022fcd91dULL,
    0x3c8a6f4144a6c38dULL, 0x3fef463b88628cd6ULL, 0x3c807a05b0e4047dULL, 0x3fef3f49917ddc96ULL,
    0x3c968efde3a8a894ULL, 0x3fef387a6e756238ULL, 0x3c875e18f274487dULL, 0x3fef31ce4fb2a63fULL,
    0x3c80472b981fe7f2ULL, 0x3fef2b4565e27cddULL, 0xbc96b87b3f71085eULL, 0x3fef24dfe1f56381ULL,
    0x3c82f7e16d09ab31ULL, 0x3fef1e9df51fdee1ULL, 0xbc3d219b1a6fbffaULL, 0x3fef187fd0dad990ULL,
    0x3c8b3782720c0ab4ULL, 0x3fef1285a6e4030bULL, 0x3c6e149289cecb8fULL, 0x3fef0cafa93e2f56ULL,
    0x3c834d754db0abb6ULL, 0x3fef06fe0a31b715ULL, 0x3c864201e2ac744cULL, 0x3fef0170fc4cd831ULL,
    0x3c8fdd395dd3f84aULL, 0x3feefc08b26416ffULL, 0xbc86a3803b8e5b04ULL, 0x3feef6c55f929ff1ULL,
    0xbc924aedcc4b5068ULL, 0x3feef1a7373aa9cbULL, 0xbc9907f81b512d8eULL, 0x3feeecae6d05d866ULL,
    0xbc71d1e83e9436d2ULL, 0x3feee7db34e59ff7ULL, 0xbc991919b3ce1b15ULL, 0x3feee32dc313a8e5ULL,
    0x3c859f48a72a4c6dULL, 0x3feedea64c123422ULL, 0xbc9312607a28698aULL, 0x3feeda4504ac801cULL,
    0xbc58a78f4817895bULL, 0x3feed60a21f72e2aULL, 0xbc7c2c9b67499a1bULL, 0x3feed1f5d950a897ULL,
    0x3c4363ed60c2ac11ULL, 0x3feece086061892dULL, 0x3c9666093b0664efULL, 0x3feeca41ed1d0057ULL,
    0x3c6ecce1daa10379ULL, 0x3feec6a2b5c13cd0ULL, 0x3c93ff8e3f0f1230ULL, 0x3feec32af0d7d3deULL,
    0x3c7690cebb7aafb0ULL, 0x3feebfdad5362a27ULL, 0x3c931dbdeb54e077ULL, 0x3feebcb299fddd0dULL,
    0xbc8f94340071a38eULL, 0x3feeb9b2769d2ca7ULL, 0xbc87deccdc93a349ULL, 0x3feeb6daa2cf6642ULL,
    0xbc78dec6bd0f385fULL, 0x3feeb42b569d4f82ULL, 0xbc861246ec7b5cf6ULL, 0x3feeb1a4ca5d920fULL,
    0x3c93350518fdd78eULL, 0x3feeaf4736b527daULL, 0x3c7b98b72f8a9b05ULL, 0x3feead12d497c7fdULL,
    0x3c9063e1e21c5409ULL, 0x3feeab07dd485429ULL, 0x3c34c7855019c6eaULL, 0x3feea9268a5946b7ULL,
    0x3c9432e62b64c035ULL, 0x3feea76f15ad2148ULL, 0xbc8ce44a6199769fULL, 0x3feea5e1b976dc09ULL,
    0xbc8c33c53bef4da8ULL, 0x3feea47eb03a5585ULL, 0xbc845378892be9aeULL, 0x3feea34634ccc320ULL,
    0xbc93cedd78565858ULL, 0x3feea23882552225ULL, 0x3c5710aa807e1964ULL, 0x3feea155d44ca973ULL,
    0xbc93b3efbf5e2228ULL, 0x3feea09e667f3bcdULL, 0xbc6a12ad8734b982ULL, 0x3feea012750bdabfULL,
    0xbc6367efb86da9eeULL, 0x3fee9fb23c651a2fULL, 0xbc80dc3d54e08851ULL, 0x3fee9f7df9519484ULL,
    0xbc781f647e5a3ecfULL, 0x3fee9f75e8ec5f74ULL, 0xbc86ee4ac08b7db0ULL, 0x3fee9f9a48a58174ULL,
    0xbc8619321e55e68aULL, 0x3fee9feb564267c9ULL, 0x3c909ccb5e09d4d3ULL, 0x3feea0694fde5d3fULL,
    0xbc7b32dcb94da51dULL, 0x3feea11473eb0187ULL, 0x3c94ecfd5467c06bULL, 0x3feea1ed0130c132ULL,
    0x3c65ebe1abd66c55ULL, 0x3feea2f336cf4e62ULL, 0xbc88a1c52fb3cf42ULL, 0x3feea427543e1a12ULL,
    0xbc9369b6f13b3734ULL, 0x3feea589994cce13ULL, 0xbc805e843a19ff1eULL, 0x3feea71a4623c7adULL,
    0xbc94d450d872576eULL, 0x3feea8d99b4492edULL, 0x3c90ad675b0e8a00ULL, 0x3feeaac7d98a6699ULL,
    0x3c8db72fc1f0eab4ULL, 0x3feeace5422aa0dbULL, 0xbc65b6609cc5e7ffULL, 0x3feeaf3216b5448cULL,
    0x3c7bf68359f35f44ULL, 0x3feeb1ae99157736ULL, 0xbc93091fa71e3d83ULL, 0x3feeb45b0b91ffc6ULL,
    0xbc5da9b88b6c1e29ULL, 0x3feeb737b0cdc5e5ULL, 0xbc6c23f97c90b959ULL, 0x3feeba44cbc8520fULL,
    0xbc92434322f4f9aaULL, 0x3feebd829fde4e50ULL, 0xbc85ca6cd7668e4bULL, 0x3feec0f170ca07baULL,
    0x3c71affc2b91ce27ULL, 0x3feec49182a3f090ULL, 0x3c6dd235e10a73bbULL, 0x3feec86319e32323ULL,
    0xbc87c50422622263ULL, 0x3feecc667b5de565ULL, 0x3c8b1c86e3e231d5ULL, 0x3feed09bec4a2d33ULL,
    0xbc91bbd1d3bcbb15ULL, 0x3feed503b23e255dULL, 0x3c90cc319cee31d2ULL, 0x3feed99e1330b358ULL,
    0x3c8469846e735ab3ULL, 0x3feede6b5579fdbfULL, 0xbc82dfcd978e9db4ULL, 0x3feee36bbfd3f37aULL,
    0x3c8c1a7792cb3387ULL, 0x3feee89f995ad3adULL, 0xbc907b8f4ad1d9faULL, 0x3feeee07298db666ULL,
    0xbc55c3d956dcaebaULL, 0x3feef3a2b84f15fbULL, 0xbc90a40e3da6f640ULL, 0x3feef9728de5593aULL,
    0xbc68d6f438ad9334ULL, 0x3feeff76f2fb5e47ULL, 0xbc91eee26b588a35ULL, 0x3fef05b030a1064aULL,
    0x3c74ffd70a5fddcdULL, 0x3fef0c1e904bc1d2ULL, 0xbc91bdfbfa9298acULL, 0x3fef12c25bd71e09ULL,
    0x3c736eae30af0cb3ULL, 0x3fef199bdd85529cULL, 0x3c8ee3325c9ffd94ULL, 0x3fef20ab5fffd07aULL,
    0x3c84e08fd10959acULL, 0x3fef27f12e57d14bULL, 0x3c63cdaf384e1a67ULL, 0x3fef2f6d9406e7b5ULL,
    0x3c676b2c6c921968ULL, 0x3fef3720dcef9069ULL, 0xbc808a1883ccb5d2ULL, 0x3fef3f0b555dc3faULL,
    0xbc8fad5d3ffffa6fULL, 0x3fef472d4a07897cULL, 0xbc900dae3875a949ULL, 0x3fef4f87080d89f2ULL,
    0x3c74a385a63d07a7ULL, 0x3fef5818dcfba487ULL, 0xbc82919e2040220fULL, 0x3fef60e316c98398ULL,
    0x3c8e5a50d5c192acULL, 0x3fef69e603db3285ULL, 0x3c843a59ac016b4bULL, 0x3fef7321f301b460ULL,
    0xbc82d52107b43e1fULL, 0x3fef7c97337b9b5fULL, 0xbc892ab93b470dc9ULL, 0x3fef864614f5a129ULL,
    0x3c74b604603a88d3ULL, 0x3fef902ee78b3ff6ULL, 0x3c83c5ec519d7271ULL, 0x3fef9a51fbc74c83ULL,
    0xbc8ff7128fd391f0ULL, 0x3fefa4afa2a490daULL, 0xbc8dae98e223747dULL, 0x3fefaf482d8e67f1ULL,
    0x3c8ec3bc41aa2008ULL, 0x3fefba1bee615a27ULL, 0x3c842b94c3a9eb32ULL, 0x3fefc52b376bba97ULL,
    0x3c8a64a931d185eeULL, 0x3fefd0765b6e4540ULL, 0xbc8e37bae43be3edULL, 0x3fefdbfdad9cbe14ULL,
    0x3c77893b4d91cd9dULL, 0x3fefe7c1819e90d8ULL, 0x3c5305c14160cc89ULL, 0x3feff3c22b8f71f1ULL};

// log: ln2 (high and low part), A[0..4] (the main polynomial), B[0..10] (the polynomial near 1),
// and {1/c, log c} of the N = 128 subintervals.
static __device__ const unsigned long long kLogConst[18] = {
    0x3fe62e42fefa3800ULL, 0x3d2ef35793c76730ULL, 0xbfe0000000000001ULL, 0x3fd555555551305bULL,
    0xbfcfffffffeb4590ULL, 0x3fc999b324f10111ULL, 0xbfc55575e506c89fULL, 0xbfe0000000000000ULL,
    0x3fd5555555555577ULL, 0xbfcffffffffffdcbULL, 0x3fc999999995dd0cULL, 0xbfc55555556745a7ULL,
    0x3fc24924a344de30ULL, 0xbfbfffffa4423d65ULL, 0x3fbc7184282ad6caULL, 0xbfb999eb43b068ffULL,
    0x3fb78182f7afd085ULL, 0xbfb5521375d145cdULL};
static __device__ const unsigned long long kLogTab[256] = {
    0x3ff734f0c3e0de9fULL, 0xbfd7cc7f79e69000ULL,
    0x3ff713786a2ce91fULL, 0xbfd76feec20d0000ULL,
    0x3ff6f26008fab5a0ULL, 0xbfd713e31351e000ULL,
    0x3ff6d1a61f138c7dULL, 0xbfd6b85b38287800ULL,
    0x3ff6b1490bc5b4d1ULL, 0xbfd65d5590807800ULL,
    0x3ff69147332f0cbaULL, 0xbfd602d076180000ULL,
    0x3ff6719f18224223ULL, 0xbfd5a8ca86909000ULL,
    0x3ff6524f99a51ed9ULL, 0xbfd54f4356035000ULL,
    0x3ff63356aa8f24c4ULL, 0xbfd4f637c36b4000ULL,
    0x3ff614b36b9ddc14ULL, 0xbfd49da7fda85000ULL,
    0x3ff5f66452c65c4cULL, 0xbfd445923989a800ULL,
    0x3ff5d867b5912c4fULL, 0xbfd3edf439b0b800ULL,
    0x3ff5babccb5b90deULL, 0xbfd396ce448f7000ULL,
    0x3ff59d61f2d91a78ULL, 0xbfd3401e17bda000ULL,
    0x3ff5805612465687ULL, 0xbfd2e9e2ef468000ULL,
    0x3ff56397cee76bd3ULL, 0xbfd2941b3830e000ULL,
    0x3ff54725e2a77f93ULL, 0xbfd23ec58cda8800ULL,
    0x3ff52aff42064583ULL, 0xbfd1e9e129279000ULL,
    0x3ff50f22dbb2bddfULL, 0xbfd1956d2b48f800ULL,
    0x3ff4f38f4734ded7ULL, 0xbfd141679ab9f800ULL,
    0x3ff4d843cfde2840ULL, 0xbfd0edd094ef9800ULL,
    0x3ff4bd3ec078a3c8ULL, 0xbfd09aa518db1000ULL,
    0x3ff4a27fc3e0258aULL, 0xbfd047e65263b800ULL,
    0x3ff4880524d48434ULL, 0xbfcfeb224586f000ULL,
    0x3ff46dce1b192d0bULL, 0xbfcf474a7517b000ULL,
    0x3ff453d9d3391854ULL, 0xbfcea4443d103000ULL,
    0x3ff43a2744b4845aULL, 0xbfce020d44e9b000ULL,
    0x3ff420b54115f8fbULL, 0xbfcd60a22977f000ULL,
    0x3ff40782da3ef4b1ULL, 0xbfccc00104959000ULL,
    0x3ff3ee8f5d57fe8fULL, 0xbfcc202956891000ULL,
    0x3ff3d5d9a00b4ce9ULL, 0xbfcb81178d811000ULL,
    0x3ff3bd60c010c12bULL, 0xbfcae2c9ccd3d000ULL,
    0x3ff3a5242b75dab8ULL, 0xbfca45402e129000ULL,
    0x3ff38d22cd9fd002ULL, 0xbfc9a877681df000ULL,
    0x3ff3755bc5847a1cULL, 0xbfc90c6d69483000ULL,
    0x3ff35dce49ad36e2ULL, 0xbfc87120a645c000ULL,
    0x3ff34679984dd440ULL, 0xbfc7d68fb4143000ULL,
    0x3ff32f5cceffcb24ULL, 0xbfc73cb83c627000ULL,
    0x3ff3187775a10d49ULL, 0xbfc6a39a9b376000ULL,
    0x3ff301c8373e3990ULL, 0xbfc60b3154b7a000ULL,
    0x3ff2eb4ebb95f841ULL, 0xbfc5737d76243000ULL,
    0x3ff2d50a0219a9d1ULL, 0xbfc4dc7b8fc23000ULL,
    0x3ff2bef9a8b7fd2aULL, 0xbfc4462c51d20000ULL,
    0x3ff2a91c7a0c1babULL, 0xbfc3b08abc830000ULL,
    0x3ff293726014b530ULL, 0xbfc31b996b490000ULL,
    0x3ff27dfa5757a1f5ULL, 0xbfc2875490a44000ULL,
    0x3ff268b39b1d3bbfULL, 0xbfc1f3b9f879a000ULL,
    0x3ff2539d838ff5bdULL, 0xbfc160c8252ca000ULL,
    0x3ff23eb7aac9083bULL, 0xbfc0ce7f57f72000ULL,
    0x3ff22a012ba940b6ULL, 0xbfc03cdc49fea000ULL,
    0x3ff2157996cc4132ULL, 0xbfbf57bdbc4b8000ULL,
    0x3ff201201dd2fc9bULL, 0xbfbe370896404000ULL,
    0x3ff1ecf4494d480bULL, 0xbfbd17983ef94000ULL,
    0x3ff1d8f5528f6569ULL, 0xbfbbf9674ed8a000ULL,
    0x3ff1c52311577e7cULL, 0xbfbadc79202f6000ULL,
    0x3ff1b17c74cb26e9ULL, 0xbfb9c0c3e7288000ULL,
    0x3ff19e010c2c1ab6ULL, 0xbfb8a646b372c000ULL,
    0x3ff18ab07bb670bdULL, 0xbfb78d01b3ac0000ULL,
    0x3ff1778a25efbcb6ULL, 0xbfb674f145380000ULL,
    0x3ff1648d354c31daULL, 0xbfb55e0e6d878000ULL,
    0x3ff151b990275fddULL, 0xbfb4485cdea1e000ULL,
    0x3ff13f0ea432d24cULL, 0xbfb333d94d6aa000ULL,
    0x3ff12c8b7210f9daULL, 0xbfb22079f8c56000ULL,
    0x3ff11a3028ecb531ULL, 0xbfb10e4698622000ULL,
    0x3ff107fbda8434afULL, 0xbfaffa6c6ad20000ULL,
    0x3ff0f5ee0f4e6bb3ULL, 0xbfadda8d4a774000ULL,
    0x3ff0e4065d2a9fceULL, 0xbfabbcece4850000ULL,
    0x3ff0d244632ca521ULL, 0xbfa9a1894012c000ULL,
    0x3ff0c0a77ce2981aULL, 0xbfa788583302c000ULL,
    0x3ff0af2f83c636d1ULL, 0xbfa5715e67d68000ULL,
    0x3ff09ddb98a01339ULL, 0xbfa35c8a49658000ULL,
    0x3ff08cabaf52e7dfULL, 0xbfa149e364154000ULL,
    0x3ff07b9f2f4e28fbULL, 0xbf9e72c082eb8000ULL,
    0x3ff06ab58c358f19ULL, 0xbf9a55f152528000ULL,
    0x3ff059eea5ecf92cULL, 0xbf963d62cf818000ULL,
    0x3ff04949cdd12c90ULL, 0xbf9228fb8caa0000ULL,
    0x3ff038c6c6f0ada9ULL, 0xbf8c317b20f90000ULL,
    0x3ff02865137932a9ULL, 0xbf8419355daa0000ULL,
    0x3ff0182427ea7348ULL, 0xbf781203c2ec0000ULL,
    0x3ff008040614b195ULL, 0xbf60040979240000ULL,
    0x3fefe01ff726fa1aULL, 0x3f6feff384900000ULL,
    0x3fefa11cc261ea74ULL, 0x3f87dc41353d0000ULL,
    0x3fef6310b081992eULL, 0x3f93cea3c4c28000ULL,
    0x3fef25f63ceeadcdULL, 0x3f9b9fc114890000ULL,
    0x3feee9c8039113e7ULL, 0x3fa1b0d8ce110000ULL,
    0x3feeae8078cbb1abULL, 0x3fa58a5bd001c000ULL,
    0x3fee741aa29d0c9bULL, 0x3fa95c8340d88000ULL,
    0x3fee3a91830a99b5ULL, 0x3fad276aef578000ULL,
    0x3fee01e009609a56ULL, 0x3fb07598e598c000ULL,
    0x3fedca01e577bb98ULL, 0x3fb253f5e30d2000ULL,
    0x3fed92f20b7c9103ULL, 0x3fb42edd8b380000ULL,
    0x3fed5cac66fb5cceULL, 0x3fb606598757c000ULL,
    0x3fed272caa5ede9dULL, 0x3fb7da76356a0000ULL,
    0x3fecf26e3e6b2ccdULL, 0x3fb9ab434e1c6000ULL,
    0x3fecbe6da2a77902ULL, 0x3fbb78c7bb0d6000ULL,
    0x3fec8b266d37086dULL, 0x3fbd431332e72000ULL,
    0x3fec5894bd5d5804ULL, 0x3fbf0a3171de6000ULL,
    0x3fec26b533bb9f8cULL, 0x3fc067152b914000ULL,
    0x3febf583eeece73fULL, 0x3fc147858292b000ULL,
    0x3febc4fd75db96c1ULL, 0x3fc2266ecdca3000ULL,
    0x3feb951e0c864a28ULL, 0x3fc303d7a6c55000ULL,
    0x3feb65e2c5ef3e2cULL, 0x3fc3dfc33c331000ULL,
    0x3feb374867c9888bULL, 0x3fc4ba366b7a8000ULL,
    0x3feb094b211d304aULL, 0x3fc5933928d1f000ULL,
    0x3feadbe885f2ef7eULL, 0x3fc66acd2418f000ULL,
    0x3feaaf1d31603da2ULL, 0x3fc740f8ec669000ULL,
    0x3fea82e63fd358a7ULL, 0x3fc815c0f51af000ULL,
    0x3fea5740ef09738bULL, 0x3fc8e92954f68000ULL,
    0x3fea2c2a90ab4b27ULL, 0x3fc9bb3602f84000ULL,
    0x3fea01a01393f2d1ULL, 0x3fca8bed1c2c0000ULL,
    0x3fe9d79f24db3c1bULL, 0x3fcb5b515c01d000ULL,
    0x3fe9ae2505c7b190ULL, 0x3fcc2967ccbcc000ULL,
    0x3fe9852ef297ce2fULL, 0x3fccf635d5486000ULL,
    0x3fe95cbaeea44b75ULL, 0x3fcdc1bd3446c000ULL,
    0x3fe934c69de74838ULL, 0x3fce8c01b8cfe000ULL,
    0x3fe90d4f2f6752e6ULL, 0x3fcf5509c0179000ULL,
    0x3fe8e6528effd79dULL, 0x3fd00e6c121fb800ULL,
    0x3fe8bfce9fcc007cULL, 0x3fd071b80e93d000ULL,
    0x3fe899c0dabec30eULL, 0x3fd0d46b9e867000ULL,
    0x3fe87427aa2317fbULL, 0x3fd13687334bd000ULL,
    0x3fe84f00acb39a08ULL, 0x3fd1980d67234800ULL,
    0x3fe82a49e8653e55ULL, 0x3fd1f8ffe0cc8000ULL,
    0x3fe8060195f40260ULL, 0x3fd2595fd7636800ULL,
    0x3fe7e22563e0a329ULL, 0x3fd2b9300914a800ULL,
    0x3fe7beb377dcb5adULL, 0x3fd3187210436000ULL,
    0x3fe79baa679725c2ULL, 0x3fd377266dec1800ULL,
    0x3fe77907f2170657ULL, 0x3fd3d54ffbaf3000ULL,
    0x3fe756cadbd6130cULL, 0x3fd432eee32fe000ULL};

// pow: ln2 (high and low part), A[0..6], and {1/c, log c, the tail of log c} of the N = 128
// subintervals.
static __device__ const unsigned long long kPowConst[9] = {
    0x3fe62e42fefa3800ULL, 0x3d2ef35793c76730ULL, 0xbfe0000000000000ULL, 0xbfe5555555555560ULL,
    0x3fe0000000000006ULL, 0x3fe999999959554eULL, 0xbfe555555529a47aULL, 0xbff2495b9b4845e9ULL,
    0x3ff0002b8b263fc3ULL};
static __device__ const unsigned long long kPowTab[384] = {
    0x3ff6a00000000000ULL, 0xbfd62c82f2b9c800ULL, 0x3cfab42428375680ULL,
    0x3ff6800000000000ULL, 0xbfd5d1bdbf580800ULL, 0xbd1ca508d8e0f720ULL,
    0x3ff6600000000000ULL, 0xbfd5767717455800ULL, 0xbd2362a4d5b6506dULL,
    0x3ff6400000000000ULL, 0xbfd51aad872df800ULL, 0xbce684e49eb067d5ULL,
    0x3ff6200000000000ULL, 0xbfd4be5f95777800ULL, 0xbd041b6993293ee0ULL,
    0x3ff6000000000000ULL, 0xbfd4618bc21c6000ULL, 0x3d13d82f484c84ccULL,
    0x3ff5e00000000000ULL, 0xbfd404308686a800ULL, 0x3cdc42f3ed820b3aULL,
    0x3ff5c00000000000ULL, 0xbfd3a64c55694800ULL, 0x3d20b1c686519460ULL,
    0x3ff5a00000000000ULL, 0xbfd347dd9a988000ULL, 0x3d25594dd4c58092ULL,
    0x3ff5800000000000ULL, 0xbfd2e8e2bae12000ULL, 0x3d267b1e99b72bd8ULL,
    0x3ff5600000000000ULL, 0xbfd2895a13de8800ULL, 0x3d15ca14b6cfb03fULL,
    0x3ff5600000000000ULL, 0xbfd2895a13de8800ULL, 0x3d15ca14b6cfb03fULL,
    0x3ff5400000000000ULL, 0xbfd22941fbcf7800ULL, 0xbd165a242853da76ULL,
    0x3ff5200000000000ULL, 0xbfd1c898c1699800ULL, 0xbd1fafbc68e75404ULL,
    0x3ff5000000000000ULL, 0xbfd1675cababa800ULL, 0x3d1f1fc63382a8f0ULL,
    0x3ff4e00000000000ULL, 0xbfd1058bf9ae4800ULL, 0xbd26a8c4fd055a66ULL,
    0x3ff4c00000000000ULL, 0xbfd0a324e2739000ULL, 0xbd0c6bee7ef4030eULL,
    0x3ff4a00000000000ULL, 0xbfd0402594b4d000ULL, 0xbcf036b89ef42d7fULL,
    0x3ff4a00000000000ULL, 0xbfd0402594b4d000ULL, 0xbcf036b89ef42d7fULL,
    0x3ff4800000000000ULL, 0xbfcfb9186d5e4000ULL, 0x3d0d572aab993c87ULL,
    0x3ff4600000000000ULL, 0xbfcef0adcbdc6000ULL, 0x3d2b26b79c86af24ULL,
    0x3ff4400000000000ULL, 0xbfce27076e2af000ULL, 0xbd172f4f543fff10ULL,
    0x3ff4200000000000ULL, 0xbfcd5c216b4fc000ULL, 0x3d21ba91bbca681bULL,
    0x3ff4000000000000ULL, 0xbfcc8ff7c79aa000ULL, 0x3d27794f689f8434ULL,
    0x3ff4000000000000ULL, 0xbfcc8ff7c79aa000ULL, 0x3d27794f689f8434ULL,
    0x3ff3e00000000000ULL, 0xbfcbc286742d9000ULL, 0x3d194eb0318bb78fULL,
    0x3ff3c00000000000ULL, 0xbfcaf3c94e80c000ULL, 0x3cba4e633fcd9066ULL,
    0x3ff3a00000000000ULL, 0xbfca23bc1fe2b000ULL, 0xbd258c64dc46c1eaULL,
    0x3ff3a00000000000ULL, 0xbfca23bc1fe2b000ULL, 0xbd258c64dc46c1eaULL,
    0x3ff3800000000000ULL, 0xbfc9525a9cf45000ULL, 0xbd2ad1d904c1d4e3ULL,
    0x3ff3600000000000ULL, 0xbfc87fa06520d000ULL, 0x3d2bbdbf7fdbfa09ULL,
    0x3ff3400000000000ULL, 0xbfc7ab890210e000ULL, 0x3d2bdb9072534a58ULL,
    0x3ff3400000000000ULL, 0xbfc7ab890210e000ULL, 0x3d2bdb9072534a58ULL,
    0x3ff3200000000000ULL, 0xbfc6d60fe719d000ULL, 0xbd10e46aa3b2e266ULL,
    0x3ff3000000000000ULL, 0xbfc5ff3070a79000ULL, 0xbd1e9e439f105039ULL,
    0x3ff3000000000000ULL, 0xbfc5ff3070a79000ULL, 0xbd1e9e439f105039ULL,
    0x3ff2e00000000000ULL, 0xbfc526e5e3a1b000ULL, 0xbd20de8b90075b8fULL,
    0x3ff2c00000000000ULL, 0xbfc44d2b6ccb8000ULL, 0x3d170cc16135783cULL,
    0x3ff2c00000000000ULL, 0xbfc44d2b6ccb8000ULL, 0x3d170cc16135783cULL,
    0x3ff2a00000000000ULL, 0xbfc371fc201e9000ULL, 0x3cf178864d27543aULL,
    0x3ff2800000000000ULL, 0xbfc29552f81ff000ULL, 0xbd248d301771c408ULL,
    0x3ff2600000000000ULL, 0xbfc1b72ad52f6000ULL, 0xbd2e80a41811a396ULL,
    0x3ff2600000000000ULL, 0xbfc1b72ad52f6000ULL, 0xbd2e80a41811a396ULL,
    0x3ff2400000000000ULL, 0xbfc0d77e7cd09000ULL, 0x3d0a699688e85bf4ULL,
    0x3ff2400000000000ULL, 0xbfc0d77e7cd09000ULL, 0x3d0a699688e85bf4ULL,
    0x3ff2200000000000ULL, 0xbfbfec9131dbe000ULL, 0xbd2575545ca333f2ULL,
    0x3ff2000000000000ULL, 0xbfbe27076e2b0000ULL, 0x3d2a342c2af0003cULL,
    0x3ff2000000000000ULL, 0xbfbe27076e2b0000ULL, 0x3d2a342c2af0003cULL,
    0x3ff1e00000000000ULL, 0xbfbc5e548f5bc000ULL, 0xbd1d0c57585fbe06ULL,
    0x3ff1c00000000000ULL, 0xbfba926d3a4ae000ULL, 0x3d253935e85baac8ULL,
    0x3ff1c00000000000ULL, 0xbfba926d3a4ae000ULL, 0x3d253935e85baac8ULL,
    0x3ff1a00000000000ULL, 0xbfb8c345d631a000ULL, 0x3d137c294d2f5668ULL,
    0x3ff1a00000000000ULL, 0xbfb8c345d631a000ULL, 0x3d137c294d2f5668ULL,
    0x3ff1800000000000ULL, 0xbfb6f0d28ae56000ULL, 0xbd269737c93373daULL,
    0x3ff1600000000000ULL, 0xbfb51b073f062000ULL, 0x3d1f025b61c65e57ULL,
    0x3ff1600000000000ULL, 0xbfb51b073f062000ULL, 0x3d1f025b61c65e57ULL,
    0x3ff1400000000000ULL, 0xbfb341d7961be000ULL, 0x3d2c5edaccf913dfULL,
    0x3ff1400000000000ULL, 0xbfb341d7961be000ULL, 0x3d2c5edaccf913dfULL,
    0x3ff1200000000000ULL, 0xbfb16536eea38000ULL, 0x3d147c5e768fa309ULL,
    0x3ff1000000000000ULL, 0xbfaf0a30c0118000ULL, 0x3d2d599e83368e91ULL,
    0x3ff1000000000000ULL, 0xbfaf0a30c0118000ULL, 0x3d2d599e83368e91ULL,
    0x3ff0e00000000000ULL, 0xbfab42dd71198000ULL, 0x3d1c827ae5d6704cULL,
    0x3ff0e00000000000ULL, 0xbfab42dd71198000ULL, 0x3d1c827ae5d6704cULL,
    0x3ff0c00000000000ULL, 0xbfa77458f632c000ULL, 0xbd2cfc4634f2a1eeULL,
    0x3ff0c00000000000ULL, 0xbfa77458f632c000ULL, 0xbd2cfc4634f2a1eeULL,
    0x3ff0a00000000000ULL, 0xbfa39e87b9fec000ULL, 0x3cf502b7f526feaaULL,
    0x3ff0a00000000000ULL, 0xbfa39e87b9fec000ULL, 0x3cf502b7f526feaaULL,
    0x3ff0800000000000ULL, 0xbf9f829b0e780000ULL, 0xbd2980267c7e09e4ULL,
    0x3ff0800000000000ULL, 0xbf9f829b0e780000ULL, 0xbd2980267c7e09e4ULL,
    0x3ff0600000000000ULL, 0xbf97b91b07d58000ULL, 0xbd288d5493faa639ULL,
    0x3ff0400000000000ULL, 0xbf8fc0a8b0fc0000ULL, 0xbcdf1e7cf6d3a69cULL,
    0x3ff0400000000000ULL, 0xbf8fc0a8b0fc0000ULL, 0xbcdf1e7cf6d3a69cULL,
    0x3ff0200000000000ULL, 0xbf7fe02a6b100000ULL, 0xbd19e23f0dda40e4ULL,
    0x3ff0200000000000ULL, 0xbf7fe02a6b100000ULL, 0xbd19e23f0dda40e4ULL,
    0x3ff0000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x3ff0000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x3fefc00000000000ULL, 0x3f80101575890000ULL, 0xbd10c76b999d2be8ULL,
    0x3fef800000000000ULL, 0x3f90205658938000ULL, 0xbd23dc5b06e2f7d2ULL,
    0x3fef400000000000ULL, 0x3f98492528c90000ULL, 0xbd2aa0ba325a0c34ULL,
    0x3fef000000000000ULL, 0x3fa0415d89e74000ULL, 0x3d0111c05cf1d753ULL,
    0x3feec00000000000ULL, 0x3fa466aed42e0000ULL, 0xbd2c167375bdfd28ULL,
    0x3fee800000000000ULL, 0x3fa894aa149fc000ULL, 0xbd197995d05a267dULL,
    0x3fee400000000000ULL, 0x3faccb73cdddc000ULL, 0xbd1a68f247d82807ULL,
    0x3fee200000000000ULL, 0x3faeea31c006c000ULL, 0xbd0e113e4fc93b7bULL,
    0x3fede00000000000ULL, 0x3fb1973bd1466000ULL, 0xbd25325d560d9e9bULL,
    0x3feda00000000000ULL, 0x3fb3bdf5a7d1e000ULL, 0x3d2cc85ea5db4ed7ULL,
    0x3fed600000000000ULL, 0x3fb5e95a4d97a000ULL, 0xbd2c69063c5d1d1eULL,
    0x3fed400000000000ULL, 0x3fb700d30aeac000ULL, 0x3cec1e8da99ded32ULL,
    0x3fed000000000000ULL, 0x3fb9335e5d594000ULL, 0x3d23115c3abd47daULL,
    0x3fecc00000000000ULL, 0x3fbb6ac88dad6000ULL, 0xbd1390802bf768e5ULL,
    0x3feca00000000000ULL, 0x3fbc885801bc4000ULL, 0x3d2646d1c65aacd3ULL,
    0x3fec600000000000ULL, 0x3fbec739830a2000ULL, 0xbd2dc068afe645e0ULL,
    0x3fec400000000000ULL, 0x3fbfe89139dbe000ULL, 0xbd2534d64fa10afdULL,
    0x3fec000000000000ULL, 0x3fc1178e8227e000ULL, 0x3d21ef78ce2d07f2ULL,
    0x3febe00000000000ULL, 0x3fc1aa2b7e23f000ULL, 0x3d2ca78e44389934ULL,
    0x3feba00000000000ULL, 0x3fc2d1610c868000ULL, 0x3d039d6ccb81b4a1ULL,
    0x3feb800000000000ULL, 0x3fc365fcb0159000ULL, 0x3cc62fa8234b7289ULL,
    0x3feb400000000000ULL, 0x3fc4913d8333b000ULL, 0x3d25837954fdb678ULL,
    0x3feb200000000000ULL, 0x3fc527e5e4a1b000ULL, 0x3d2633e8e5697dc7ULL,
    0x3feae00000000000ULL, 0x3fc6574ebe8c1000ULL, 0x3d19cf8b2c3c2e78ULL,
    0x3feac00000000000ULL, 0x3fc6f0128b757000ULL, 0xbd25118de59c21e1ULL,
    0x3feaa00000000000ULL, 0x3fc7898d85445000ULL, 0xbd1c661070914305ULL,
    0x3fea600000000000ULL, 0x3fc8beafeb390000ULL, 0xbd073d54aae92cd1ULL,
    0x3fea400000000000ULL, 0x3fc95a5adcf70000ULL, 0x3d07f22858a0ff6fULL,
    0x3fea000000000000ULL, 0x3fca93ed3c8ae000ULL, 0xbd28724350562169ULL,
    0x3fe9e00000000000ULL, 0x3fcb31d8575bd000ULL, 0xbd0c358d4eace1aaULL,
    0x3fe9c00000000000ULL, 0x3fcbd087383be000ULL, 0xbd2d4bc4595412b6ULL,
    0x3fe9a00000000000ULL, 0x3fcc6ffbc6f01000ULL, 0xbcf1ec72c5962bd2ULL,
    0x3fe9600000000000ULL, 0x3fcdb13db0d49000ULL, 0xbd2aff2af715b035ULL,
    0x3fe9400000000000ULL, 0x3fce530effe71000ULL, 0x3cc212276041f430ULL,
    0x3fe9200000000000ULL, 0x3fcef5ade4dd0000ULL, 0xbcca211565bb8e11ULL,
    0x3fe9000000000000ULL, 0x3fcf991c6cb3b000ULL, 0x3d1bcbecca0cdf30ULL,
    0x3fe8c00000000000ULL, 0x3fd07138604d5800ULL, 0x3cf89cdb16ed4e91ULL,
    0x3fe8a00000000000ULL, 0x3fd0c42d67616000ULL, 0x3d27188b163ceae9ULL,
    0x3fe8800000000000ULL, 0x3fd1178e8227e800ULL, 0xbd2c210e63a5f01cULL,
    0x3fe8600000000000ULL, 0x3fd16b5ccbacf800ULL, 0x3d2b9acdf7a51681ULL,
    0x3fe8400000000000ULL, 0x3fd1bf99635a6800ULL, 0x3d2ca6ed5147bdb7ULL,
    0x3fe8200000000000ULL, 0x3fd214456d0eb800ULL, 0x3d0a87deba46baeaULL,
    0x3fe7e00000000000ULL, 0x3fd2bef07cdc9000ULL, 0x3d2a9cfa4a5004f4ULL,
    0x3fe7c00000000000ULL, 0x3fd314f1e1d36000ULL, 0xbd28e27ad3213cb8ULL,
    0x3fe7a00000000000ULL, 0x3fd36b6776be1000ULL, 0x3d116ecdb0f177c8ULL,
    0x3fe7800000000000ULL, 0x3fd3c25277333000ULL, 0x3d183b54b606bd5cULL,
    0x3fe7600000000000ULL, 0x3fd419b423d5e800ULL, 0x3d08e436ec90e09dULL,
    0x3fe7400000000000ULL, 0x3fd4718dc271c800ULL, 0xbd2f27ce0967d675ULL,
    0x3fe7200000000000ULL, 0x3fd4c9e09e173000ULL, 0xbd2e20891b0ad8a4ULL,
    0x3fe7000000000000ULL, 0x3fd522ae0738a000ULL, 0x3d2ebe708164c759ULL,
    0x3fe6e00000000000ULL, 0x3fd57bf753c8d000ULL, 0x3d1fadedee5d40efULL,
    0x3fe6c00000000000ULL, 0x3fd5d5bddf596000ULL, 0xbd0a0b2a08a465dcULL};

constexpr unsigned long long kOneBits = 0x3ff0000000000000ULL;
constexpr unsigned long long kInfBits = 0x7ff0000000000000ULL;
constexpr unsigned long long kSignBit = 0x8000000000000000ULL;
constexpr unsigned long long kQuietBit = 0x0008000000000000ULL;
constexpr unsigned long long kDefaultNaN = 0xfff8000000000000ULL;  // the x86 default NaN
constexpr unsigned int kSignBias = 0x800U << 7;                     // pow: a negative result

__device__ __forceinline__ unsigned long long bits(const double x) {
  return static_cast<unsigned long long>(__double_as_longlong(x));
}

__device__ __forceinline__ double from_bits(const unsigned long long u) {
  return __longlong_as_double(static_cast<long long>(u));
}

__device__ __forceinline__ double table(const unsigned long long* t, const int i) {
  return from_bits(__ldg(t + i));
}

__device__ __forceinline__ bool is_nan(const double x) {
  return (bits(x) << 1) > (kInfBits << 1);
}

__device__ __forceinline__ double quiet(const double x) { return from_bits(bits(x) | kQuietBit); }

__device__ __forceinline__ double negate(const double x) { return from_bits(bits(x) ^ kSignBit); }

// The x86 rules for an operation with a NaN operand: the first NaN operand, quieted.
__device__ __forceinline__ double add_x86(const double a, const double b) {
  if (is_nan(a)) {
    return quiet(a);
  }
  if (is_nan(b)) {
    return quiet(b);
  }
  const double s = __dadd_rn(a, b);
  return is_nan(s) ? from_bits(kDefaultNaN) : s;  // inf + -inf
}

__device__ __forceinline__ double mul_x86(const double a, const double b) {
  if (is_nan(a)) {
    return quiet(a);
  }
  if (is_nan(b)) {
    return quiet(b);
  }
  const double p = __dmul_rn(a, b);
  return is_nan(p) ? from_bits(kDefaultNaN) : p;  // 0 * inf
}

__device__ __forceinline__ double div_x86(const double a, const double b) {
  if (is_nan(a)) {
    return quiet(a);
  }
  if (is_nan(b)) {
    return quiet(b);
  }
  const double q = __ddiv_rn(a, b);
  return is_nan(q) ? from_bits(kDefaultNaN) : q;  // 0 / 0, inf / inf
}

// (x - x) / (x - x): the input NaN quieted, else the default NaN.
__device__ __forceinline__ double invalid(const double x) {
  return is_nan(x) ? quiet(x) : from_bits(kDefaultNaN);
}

// exp(x) = scale (1 + tmp) near the overflow (k > 0) and in the subnormal range (k < 0).
__device__ __forceinline__ double exp_specialcase(const double tmp, unsigned long long sbits,
                                                  const unsigned long long ki) {
  if ((ki & 0x80000000ULL) == 0) {
    sbits -= 1009ULL << 52;
    const double scale = from_bits(sbits);
    return __dmul_rn(__fma_rn(scale, tmp, scale), 0x1p1009);
  }
  sbits += 1022ULL << 52;
  const double scale = from_bits(sbits);
  const double st = __dmul_rn(tmp, scale);
  double y = __dadd_rn(scale, st);
  if (1.0 > y) {
    const double hi = __dadd_rn(y, 1.0);
    const double lo = __dadd_rn(__dsub_rn(scale, y), st);
    double t = __dadd_rn(__dsub_rn(1.0, hi), y);
    t = __dadd_rn(t, lo);
    t = __dadd_rn(t, hi);
    y = __dsub_rn(t, 1.0);
    if (y == 0.0) {
      y = 0.0;
    }
  }
  return __dmul_rn(y, 0x1p-1022);
}

// The same for pow, whose scale may carry the sign of the result.
__device__ __forceinline__ double pow_specialcase(const double tmp, unsigned long long sbits,
                                                  const unsigned long long ki) {
  if ((ki & 0x80000000ULL) == 0) {
    sbits -= 1009ULL << 52;
    const double scale = from_bits(sbits);
    return __dmul_rn(__fma_rn(scale, tmp, scale), 0x1p1009);
  }
  sbits += 1022ULL << 52;
  const double scale = from_bits(sbits);
  const double st = __dmul_rn(tmp, scale);
  double y = __dadd_rn(scale, st);
  if (1.0 > fabs(y)) {
    const double lo = __dadd_rn(__dsub_rn(scale, y), st);
    const double one = (y < 0.0) ? -1.0 : 1.0;
    const double hi = __dadd_rn(y, one);
    double t = __dadd_rn(__dsub_rn(one, hi), y);
    t = __dadd_rn(t, lo);
    t = __dadd_rn(t, hi);
    y = __dsub_rn(t, one);
    if (y == 0.0) {
      y = from_bits(sbits & kSignBit);
    }
  }
  return __dmul_rn(y, 0x1p-1022);
}

// The exp of pow: exp(x + xtail) with the sign given by sign_bias.
__device__ __forceinline__ double pow_exp(const double x, const double xtail,
                                          const unsigned int sign_bias) {
  unsigned int abstop = static_cast<unsigned int>(bits(x) >> 52) & 0x7ffU;
  if (abstop - 0x3c9U > 0x3eU) {
    if (static_cast<int>(abstop - 0x3c9U) < 0) {
      const double one = __dadd_rn(x, 1.0);
      return sign_bias != 0U ? negate(one) : one;
    }
    if (abstop >= 0x409U) {
      // __math_uflow / __math_oflow: (+-0x1p-767) * 0x1p-767, (+-0x1p769) * 0x1p769
      const bool negative = sign_bias != 0U;
      if ((bits(x) >> 63) != 0) {
        return negative ? -0.0 : 0.0;
      }
      return negative ? -CUDART_INF : CUDART_INF;
    }
    abstop = 0U;  // the large x below
  }
  const double kd0 = __fma_rn(x, from_bits(kExpConst[0]), from_bits(kExpConst[1]));
  const unsigned long long ki = bits(kd0);
  const double kd = __dsub_rn(kd0, from_bits(kExpConst[1]));
  const double r1 = __fma_rn(kd, from_bits(kExpConst[2]), x);
  const double r0 = __fma_rn(kd, from_bits(kExpConst[3]), r1);
  const int idx = 2 * static_cast<int>(ki & 127ULL);
  const unsigned long long top = (ki + sign_bias) << 45;
  const double tail = table(kExpTab, idx);
  const unsigned long long sbits = __ldg(kExpTab + idx + 1) + top;
  const double r = __dadd_rn(xtail, r0);
  const double p23 = __fma_rn(r, from_bits(kExpConst[5]), from_bits(kExpConst[4]));
  const double tr = __dadd_rn(r, tail);
  const double r2 = __dmul_rn(r, r);
  const double p45 = __fma_rn(r, from_bits(kExpConst[7]), from_bits(kExpConst[6]));
  const double t1 = __fma_rn(p23, r2, tr);
  const double r4 = __dmul_rn(r2, r2);
  const double tmp = __fma_rn(r4, p45, t1);
  if (abstop == 0U) {
    return pow_specialcase(tmp, sbits, ki);
  }
  const double scale = from_bits(sbits);
  return __fma_rn(scale, tmp, scale);
}

// The log of pow: log(x) = y + *tail with about 15 bits beyond double precision. ix: the bits of
// a positive normal x (a subnormal one normalized with a negative exponent).
__device__ __forceinline__ double pow_log(const unsigned long long ix, double* tail) {
  const unsigned long long tmp = ix - 0x3fe6955500000000ULL;
  const int i = static_cast<int>((tmp >> 45) & 127ULL);
  const long long k = static_cast<long long>(tmp) >> 52;
  const unsigned long long iz = ix - (tmp & (0xfffULL << 52));
  const double z = from_bits(iz);
  const double kd = static_cast<double>(static_cast<int>(k));
  const double invc = table(kPowTab, 3 * i);
  const double logc = table(kPowTab, 3 * i + 1);
  const double logctail = table(kPowTab, 3 * i + 2);
  const double ln2hi = from_bits(kPowConst[0]);
  const double ln2lo = from_bits(kPowConst[1]);
  const double t1 = __fma_rn(kd, ln2hi, logc);
  const double lo1 = __fma_rn(kd, ln2lo, logctail);
  const double r = __fma_rn(z, invc, -1.0);
  const double ar = __dmul_rn(r, from_bits(kPowConst[2]));
  const double q12 = __fma_rn(r, from_bits(kPowConst[4]), from_bits(kPowConst[3]));
  const double q34 = __fma_rn(r, from_bits(kPowConst[6]), from_bits(kPowConst[5]));
  const double t2 = __dadd_rn(r, t1);
  const double lo2 = __dadd_rn(__dsub_rn(t1, t2), r);
  const double ar2 = __dmul_rn(r, ar);
  const double ar3 = __dmul_rn(r, ar2);
  const double lo3 = __fma_rn(ar, r, -ar2);
  const double hi = __dadd_rn(t2, ar2);
  const double q56 = __fma_rn(r, from_bits(kPowConst[8]), from_bits(kPowConst[7]));
  const double w34 = __fma_rn(q56, ar2, q34);
  const double lo4 = __dadd_rn(__dsub_rn(t2, hi), ar2);
  const double w12 = __fma_rn(ar2, w34, q12);
  double s = __dadd_rn(lo1, lo2);
  s = __dadd_rn(s, lo3);
  s = __dadd_rn(s, lo4);
  const double lo = __fma_rn(ar3, w12, s);
  const double y = __dadd_rn(hi, lo);
  *tail = __dadd_rn(__dsub_rn(hi, y), lo);
  return y;
}

// 0: y is not an integer, 1: an odd integer, 2: an even integer (iy: a nonzero finite y).
__device__ __forceinline__ int checkint(const unsigned long long iy) {
  const int e = static_cast<int>((iy >> 52) & 0x7ffULL);
  if (e < 0x3ff) {
    return 0;
  }
  if (e > 0x3ff + 52) {
    return 2;
  }
  if ((iy & ((1ULL << (0x3ff + 52 - e)) - 1ULL)) != 0ULL) {
    return 0;
  }
  if ((iy & (1ULL << (0x3ff + 52 - e))) != 0ULL) {
    return 1;
  }
  return 2;
}

__device__ __forceinline__ bool zeroinfnan(const unsigned long long i) {
  return 2ULL * i - 1ULL >= 2ULL * kInfBits - 1ULL;
}

__device__ __forceinline__ bool is_signaling(const double x) {
  return 2ULL * (bits(x) ^ kQuietBit) > 2ULL * 0x7ff8000000000000ULL;
}

}  // namespace detail

__device__ inline double exp(const double x) {
  using namespace detail;
  const unsigned long long ix = bits(x);
  unsigned int abstop = static_cast<unsigned int>(ix >> 52) & 0x7ffU;
  if (abstop - 0x3c9U > 0x3eU) {
    if (static_cast<int>(abstop - 0x3c9U) < 0) {
      return add_x86(x, 1.0);  // |x| < 2^-54
    }
    if (abstop >= 0x409U) {
      if (ix == 0xfff0000000000000ULL) {
        return 0.0;
      }
      if (abstop >= 0x7ffU) {
        return add_x86(x, 1.0);  // +inf, NaN
      }
      return (ix >> 63) != 0 ? 0.0 : CUDART_INF;  // __math_uflow (0), __math_oflow (0)
    }
    abstop = 0U;  // the large x below
  }
  const double shift = from_bits(kExpConst[1]);
  const double kd0 = __fma_rn(x, from_bits(kExpConst[0]), shift);
  const unsigned long long ki = bits(kd0);
  const double kd = __dsub_rn(kd0, shift);
  const double r1 = __fma_rn(kd, from_bits(kExpConst[2]), x);
  const double r = __fma_rn(kd, from_bits(kExpConst[3]), r1);
  const int idx = 2 * static_cast<int>(ki & 127ULL);
  const unsigned long long top = ki << 45;
  const double tail = table(kExpTab, idx);
  const unsigned long long sbits = __ldg(kExpTab + idx + 1) + top;
  const double p23 = __fma_rn(r, from_bits(kExpConst[5]), from_bits(kExpConst[4]));
  const double tr = __dadd_rn(r, tail);
  const double r2 = __dmul_rn(r, r);
  const double p45 = __fma_rn(r, from_bits(kExpConst[7]), from_bits(kExpConst[6]));
  const double t1 = __fma_rn(p23, r2, tr);
  const double r4 = __dmul_rn(r2, r2);
  const double tmp = __fma_rn(r4, p45, t1);
  if (abstop == 0U) {
    return exp_specialcase(tmp, sbits, ki);
  }
  const double scale = from_bits(sbits);
  return __fma_rn(scale, tmp, scale);
}

__device__ inline double log(const double x) {
  using namespace detail;
  unsigned long long ix = bits(x);
  const unsigned int top = static_cast<unsigned int>(ix >> 48);
  if (ix - 0x3fee000000000000ULL <= 0x308ffffffffffULL) {
    // |x - 1| small: 1 - 2^-4 <= x < 1 + 0x1.09p-4
    if (ix == kOneBits) {
      return 0.0;
    }
    const double r = __dsub_rn(x, 1.0);
    const double u12 = __fma_rn(r, from_bits(kLogConst[9]), from_bits(kLogConst[8]));
    const double u45 = __fma_rn(r, from_bits(kLogConst[12]), from_bits(kLogConst[11]));
    const double r2 = __dmul_rn(r, r);
    const double u78 = __fma_rn(r, from_bits(kLogConst[15]), from_bits(kLogConst[14]));
    const double v123 = __fma_rn(r2, from_bits(kLogConst[10]), u12);
    const double v456 = __fma_rn(r2, from_bits(kLogConst[13]), u45);
    const double r3 = __dmul_rn(r, r2);
    const double v789 = __fma_rn(r2, from_bits(kLogConst[16]), u78);
    const double w7 = __fma_rn(r3, from_bits(kLogConst[17]), v789);
    const double w4 = __fma_rn(w7, r3, v456);
    const double s = __fma_rn(w4, r3, v123);
    const double t1 = __fma_rn(r, 0x1p27, r);
    const double rhi = __fma_rn(-0x1p27, r, t1);
    const double b0 = from_bits(kLogConst[7]);
    const double rhi2 = __dmul_rn(rhi, rhi);
    const double rlo = __dsub_rn(r, rhi);
    const double hi = __fma_rn(rhi2, b0, r);
    const double lo = __fma_rn(rhi2, b0, __dsub_rn(r, hi));
    const double lo2 = __fma_rn(__dmul_rn(b0, rlo), __dadd_rn(r, rhi), lo);
    const double y1 = __fma_rn(s, r3, lo2);
    return __dadd_rn(hi, y1);
  }
  if (top - 0x0010U > 0x7fdfU) {
    // x < 2^-1022, inf or NaN
    if ((ix << 1) == 0ULL) {
      return -CUDART_INF;  // __math_divzero (1)
    }
    if (ix == kInfBits) {
      return x;
    }
    if ((top & 0x8000U) != 0U || (top & 0x7ff0U) == 0x7ff0U) {
      return invalid(x);
    }
    ix = bits(__dmul_rn(x, 0x1p52));  // a subnormal x, normalized
    ix -= 52ULL << 52;
  }
  const unsigned long long tmp = ix - 0x3fe6000000000000ULL;
  const int i = static_cast<int>((tmp >> 45) & 127ULL);
  const long long k = static_cast<long long>(tmp) >> 52;
  const unsigned long long iz = ix - (tmp & (0xfffULL << 52));
  const double invc = table(kLogTab, 2 * i);
  const double logc = table(kLogTab, 2 * i + 1);
  const double z = from_bits(iz);
  const double kd = static_cast<double>(static_cast<int>(k));
  const double w = __fma_rn(kd, from_bits(kLogConst[0]), logc);
  const double r = __fma_rn(z, invc, -1.0);
  const double p12 = __fma_rn(r, from_bits(kLogConst[4]), from_bits(kLogConst[3]));
  const double hi = __dadd_rn(r, w);
  const double r2 = __dmul_rn(r, r);
  const double lo = __fma_rn(kd, from_bits(kLogConst[1]), __dadd_rn(__dsub_rn(w, hi), r));
  const double r3 = __dmul_rn(r, r2);
  const double p34 = __fma_rn(r, from_bits(kLogConst[6]), from_bits(kLogConst[5]));
  const double t = __fma_rn(r2, from_bits(kLogConst[2]), lo);
  const double q = __fma_rn(p34, r2, p12);
  const double y1 = __fma_rn(r3, q, t);
  return __dadd_rn(y1, hi);
}

__device__ inline double pow(const double x, const double y) {
  using namespace detail;
  unsigned int sign_bias = 0U;
  unsigned long long ix = bits(x);
  const unsigned long long iy = bits(y);
  unsigned int topx = static_cast<unsigned int>(ix >> 52);
  const unsigned int topy = static_cast<unsigned int>(iy >> 52);
  if (topx - 0x001U >= 0x7ffU - 0x001U || (topy & 0x7ffU) - 0x3beU >= 0x43eU - 0x3beU) {
    // x < 2^-1022, inf or NaN, or |y| < 2^-65, |y| >= 2^63 or NaN
    if (zeroinfnan(iy)) {
      if (2ULL * iy == 0ULL) {
        return is_signaling(x) ? add_x86(x, y) : 1.0;
      }
      if (ix == kOneBits) {
        return is_signaling(y) ? add_x86(x, y) : 1.0;
      }
      if (2ULL * ix > 2ULL * kInfBits || 2ULL * iy > 2ULL * kInfBits) {
        return add_x86(x, y);
      }
      if (2ULL * ix == 2ULL * kOneBits) {
        return 1.0;
      }
      if ((2ULL * ix < 2ULL * kOneBits) == ((iy >> 63) == 0ULL)) {
        return 0.0;  // |x| < 1 and y = inf, or |x| > 1 and y = -inf
      }
      return __dmul_rn(y, y);
    }
    if (zeroinfnan(ix)) {
      double x2 = mul_x86(x, x);
      unsigned int negative = 0U;
      if ((ix >> 63) != 0ULL && checkint(iy) == 1) {
        x2 = negate(x2);
        negative = 1U;
      }
      if (2ULL * ix == 0ULL && (iy >> 63) != 0ULL) {
        return negative != 0U ? -CUDART_INF : CUDART_INF;  // __math_divzero
      }
      return (iy >> 63) != 0ULL ? div_x86(1.0, x2) : x2;
    }
    // x and y nonzero and finite
    if ((ix >> 63) != 0ULL) {
      const int yint = checkint(iy);
      if (yint == 0) {
        return invalid(x);
      }
      if (yint == 1) {
        sign_bias = kSignBias;
      }
      ix &= 0x7fffffffffffffffULL;
      topx &= 0x7ffU;
    }
    if ((topy & 0x7ffU) - 0x3beU >= 0x43eU - 0x3beU) {
      if (ix == kOneBits) {
        return 1.0;
      }
      if ((topy & 0x7ffU) < 0x3beU) {
        return ix > kOneBits ? __dadd_rn(1.0, y) : __dsub_rn(1.0, y);  // |y| < 2^-65
      }
      return ((ix > kOneBits) == (topy < 0x800U)) ? CUDART_INF : 0.0;  // overflow, underflow
    }
    if (topx == 0U) {
      ix = bits(__dmul_rn(x, 0x1p52));  // a subnormal x, normalized
      ix &= 0x7fffffffffffffffULL;
      ix -= 52ULL << 52;
    }
  }
  double lo = 0.0;
  const double hi = pow_log(ix, &lo);
  const double ehi = __dmul_rn(y, hi);
  const double elo = __fma_rn(y, lo, __fma_rn(y, hi, -ehi));
  return pow_exp(ehi, elo, sign_bias);
}

}  // namespace tenryu::core::glibc_libm
