#pragma once
#include<iostream>

#define MY_ASSERT(exp) if(!(exp)) std::cerr<<"error at: "<<__FILE__<<": "<<__LINE__<<std::endl;