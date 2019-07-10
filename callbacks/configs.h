#pragma once

namespace intgemm {
namespace callbacks {

struct Dummy {
};

struct Unquantize {
  float unquant_mult;

  Unquantize(float unquant_mult) : unquant_mult(unquant_mult) {}
};

struct Write {
  float* addr;

  Write(float* addr) : addr(addr) {}
};

struct AddBias {
  const float* bias_addr;

  AddBias(const float* bias_addr) : bias_addr(bias_addr) {}
};

struct UnquantizeAndWrite {
  float unquant_mult;
  float* addr;

  UnquantizeAndWrite(float unquant_mult, float* addr) : unquant_mult(unquant_mult), addr(addr) {}
};

struct UnquantizeAndAddBiasAndWrite {
  float unquant_mult;
  const float* bias_addr;
  float* output_addr;

  UnquantizeAndAddBiasAndWrite(float unquant_mult, const float* bias_addr, float* output_addr) : unquant_mult(unquant_mult), bias_addr(bias_addr), output_addr(output_addr) {}
};

}
}
