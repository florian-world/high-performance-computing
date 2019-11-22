#pragma once

#include <array>
#include <cassert>
#include <cmath>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

static constexpr size_t alignment = 128; // align arrays with cache lines

#ifdef SINGLE_PRECISION
using value_t = float;
#define MPI_VALUE_T MPI_FLOAT
#else
using value_t = double;
#define MPI_VALUE_T MPI_DOUBLE
#endif

class ArrayOfParticles
{
    size_t m_size = 0;
    size_t m_max_size = 0;
    // 5 fields: x, y, vx, vy, gamma, all initially not allocated (nullptr)
    std::array<value_t*, 5> fields = {nullptr};

    void m_allocate(size_t n_elements)
    {
        if(n_elements > m_max_size)
        {
            m_max_size = n_elements;
            for(auto & field : fields)
            {
                if(field not_eq nullptr) free(field);
                const size_t byte_size = m_max_size * sizeof(value_t);
                #ifdef _WIN32
                  field = _aligned_malloc(byte_size, alignment);
                #else
                  posix_memalign((void**) &field, alignment, byte_size);
                #endif
            }
        }
    }

public:

    ArrayOfParticles(const size_t size)
    {
        resize(size);
    }

    ~ArrayOfParticles()
    {
        for(auto & field : fields) if(field not_eq nullptr) free(field);
    }

    // move constructor and move assignment (no memory is being duplicated)
    ArrayOfParticles(ArrayOfParticles && p)
    {
        m_size = p.m_size;
        m_max_size = p.m_max_size;
        for(int i = 0; i<5; ++i) std::swap(fields[i], p.fields[i]);
    }
    ArrayOfParticles& operator=(ArrayOfParticles && p)
    {
        m_size = p.m_size;
        m_max_size = p.m_max_size;
        for(int i = 0; i<5; ++i) std::swap(fields[i], p.fields[i]);
        return * this;
    }

    // copy constructor and copy assignment are disabled:
    ArrayOfParticles(const ArrayOfParticles &p) = delete;
    ArrayOfParticles& operator=(const ArrayOfParticles &p) = delete;

    void resize(const size_t n_elements)
    {
        m_size = n_elements;
        m_allocate(n_elements);
    }

    size_t size() const {
        return m_size;
    }

    // read access pointers (to use as MPI send buffers):
    value_t* pos_x() const {
        return fields[0];
    }
    value_t* pos_y() const {
        return fields[1];
    }
    value_t* vel_x() const {
        return fields[2];
    }
    value_t* vel_y() const {
        return fields[3];
    }
    value_t* gamma() const {
        return fields[4];
    }

    // read access values:
    const value_t& pos_x(const size_t i) const {
        assert(m_size > i);
        return fields[0][i];
    }
    const value_t& pos_y(const size_t i) const {
        assert(m_size > i);
        return fields[1][i];
    }
    const value_t& vel_x(const size_t i) const {
        assert(m_size > i);
        return fields[2][i];
    }
    const value_t& vel_y(const size_t i) const {
        assert(m_size > i);
        return fields[3][i];
    }
    const value_t& gamma(const size_t i) const {
        assert(m_size > i);
        return fields[4][i];
    }

    // write access values:
    value_t& pos_x(const size_t i) {
        assert(m_size > i);
        return fields[0][i];
    }
    value_t& pos_y(const size_t i) {
        assert(m_size > i);
        return fields[1][i];
    }
    value_t& vel_x(const size_t i) {
        assert(m_size > i);
        return fields[2][i];
    }
    value_t& vel_y(const size_t i) {
        assert(m_size > i);
        return fields[3][i];
    }
    value_t& gamma(const size_t i) {
        assert(m_size > i);
        return fields[4][i];
    }

    std::string convert2csv()
    {
        // each line csv line contains 5 floats (represented each
        // by 8 to 10 chars), 4 commas, one return:
        const size_t csv_file_n_chars_hint = 55 * m_size;
        std::string csv_content;
        csv_content.reserve(csv_file_n_chars_hint);
        for(size_t i = 0; i < m_size; ++i)
            csv_content += std::to_string(pos_x(i)) + "," +
                           std::to_string(pos_y(i)) + "," +
                           std::to_string(vel_x(i)) + "," +
                           std::to_string(vel_y(i)) + "," +
                           std::to_string(gamma(i)) + "\n";

        return csv_content;
    }
};
