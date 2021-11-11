


BLISFLAGS := -I/home/adcastel/opt/blis-clang/include/blis/ -L/home/adcastel/opt/blis-clang/lib -lblis -lm 
OMPFLAGS := 
CC := gcc-10 #-mcpu=cortex-a57
CC := clang #-mcpu=cortex-a57
CFLAGS := -O3 -mtune=cortex-a57 -march=armv8.2a+fp+simd -mfpu=neon -mcpu=cortex-a57
#-mattr=+v8.2a,+fp-armv8,+neon #-Wall -O3 #-Wall  

OBJECTS := 


all: driver 
ifeq ($(DOUBLE), 1)
    CFLAGS += -DFP64
else
    CFLAGS += -DFP32
endif

driver: gemm_blis_neon_fp32.o
	$(CC) $(CFLAGS) main.c -o test_uk_blis gemm_blis_neon_fp32.o $(BLISFLAGS) $(OMPFLAGS) 


.c.o:
	        $(CC) $(CFLAGS)  -c $*.c


clean:
	rm *.o test_uk_blis


