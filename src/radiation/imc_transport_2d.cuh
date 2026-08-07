#pragma once

#include "radiation/imc_transport_persistent.cuh"

namespace tenryu::radiation {

void imc_transport_2d_persistent_cuda(const TransportInputs& in);
void imc_transport_2d_persistent_cuda(const TransportInputs& in,
                                      const tenryu::parallel::PartitionInfo& part);

}  // namespace tenryu::radiation
