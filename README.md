## Description

This repo contains a few functions to cast a buffer on a device.
It provides two main routines:
- casting: that requires to provide all parameters
- casting_iface: that requires less parameters and does some checks.

This code consider the input buffer as chunks and compress one chunk at a time.

### Seom details
The user needs to provide a buffer with enough space.
The user also can provide a memory space where a flag will be raised.
The blocks are compressed one at a time and the flag is increased accordingly.
It means when the value of the flag is four, the first four blocks have been compressed.

## Install

To compile, do
```
make
```

To create the library, do
```
make lib
```
