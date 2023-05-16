

BLISFLAGS := -I/home/adcastel/opt/blis/include/blis/ -L/home/adcastel/opt/blis/lib -lblis -lm 
OMPFLAGS := 
CC := gcc-10 #-mcpu=cortex-a57
#CC := clang #-mcpu=cortex-a57
CFLAGS := -O3 -mcpu=cortex-a57  
#CFLAGS := -O3 -mtune=cortex-a57 -march=armv8.2a+fp+simd -mfpu=neon -mcpu=cortex-a57

OBJECTS := 


all: driver 
ifeq ($(DOUBLE), 1)
    CFLAGS += -DFP64
else
    CFLAGS += -DFP32
endif

driver: gemm_blis_neon_fp32.o uk.o
	$(CC) $(CFLAGS) main.c -o test_uk_blis gemm_blis_neon_fp32.o uk.o $(BLISFLAGS) $(OMPFLAGS) 


.c.o:
	        $(CC) $(CFLAGS)  -c $*.c


clean:
	rm *.o test_uk_blis


