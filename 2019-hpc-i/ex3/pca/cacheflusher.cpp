#include "cacheflusher.h"

void CacheFlusher::flush()
{
    for (Size i = 0; i < size; ++i)
        buf[i] += i;
}
