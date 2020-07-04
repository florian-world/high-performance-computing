#ifndef CACHEFLUSHER_H
#define CACHEFLUSHER_H

class CacheFlusher
{
public:
    using Size = long long int;
    CacheFlusher(Size size = 1<<26 /* 64 MB */) :
            size(size),
            buf(new volatile unsigned char[size])
        {}


    ~CacheFlusher()
    {
        delete[] buf;
    }

    void flush();

private:
    const Size size {0};
    volatile unsigned char *buf;
};

#endif // CACHEFLUSHER_H
