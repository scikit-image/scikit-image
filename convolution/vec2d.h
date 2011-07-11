/*

SSE optimized 2D vector  
Copyright (C) 2007 YURIY V. CHESNOKOV

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


You may contact the author by e-mail chesnokov_yuriy@mail.ru

*/
#include <xmmintrin.h>
#include <mm3dnow.h>

#ifndef VEC2D_H
#define VEC2D_H
#ifndef __WIN32__
#define __declspec(x)
#endif

struct RECT {
  long left;
  long top;
  long right;
  long bottom;
};

class vec2D       //16 byte aligned
{
public:
        vec2D(const wchar_t* file);
        vec2D(unsigned int ysize = 1, unsigned int xsize = 1,
              int yoffset = 0, int xoffset = 0, const float* data = 0);           // NxM vector,  data copied in [row1][row2] ... [rowN] form
        vec2D(const vec2D& v);
        ~vec2D();

// Operators
        inline const vec2D& operator=(const vec2D& v);                             //this = v        refs
        inline const vec2D& operator=(const float* pf);                            //this = pf
        inline float& operator()(int y, int x);                                    //operator v(y,x) = v.m_data[y][x]
        inline float operator()(int y, int x) const;                               //const operator v(y,x) = v.m_data[y][x]
        inline float& operator[](unsigned int i);                                  //operator v[i] = v.m_data[y][x]  0-offset operator MATLAB order like
        inline float operator[](unsigned int i) const;                             //const operator v[i] = v.m_data[y][x]  0-offset operator MATLAB order like

// Operations
        void print(const wchar_t* file = 0) const;                                 //dump contents
        inline float get(int y, int x) const;                                      //get periodic boundary conditions
        void set(float scalar);                                                    //set all array to a scalar
        void set(float scalar, RECT& r);                                           //set selected rect in array to a scalar [ left, right ) sse 0-offset
        void setrand();                                                            //set all array to rand values
        bool copy(const vec2D& v, int left = 0, int top = 0);                      //copy v's (this_size)[hxw] region from dx,dy offset to this 0-offset  [v>=this in size]

        //0-offset array operations///////////////////////////////////////////xfisrt,yfirst=0 xlast,ylast=width-1,height-1
        inline const vec2D& operator*(const vec2D& b) const;                //new c = this*b      refs
        void add(const vec2D& a, const vec2D& b);                           //this = a.+b   sse optimized
        void sub(const vec2D& a, const vec2D& b);                           //this = a.-b   sse optimized
        void mul(const vec2D& a, const vec2D& b);                           //this = a*b
        void mult(const vec2D& a, const vec2D& b);                          //this = a*b'   sse optimized
        void div(const vec2D& a, const vec2D& b);                           //this = a./b   sse optimized
        void mule(const vec2D& a, const vec2D& b);                          //this = a.*b   sse optimized
        void add(float scalar);                                             //this = this.+scalar   sse optimized
        void sub(float scalar);                                             //this = this.-scalar   sse optimized
        void mul(float scalar);                                             //this = this.*scalar   sse optimized
        void div(float scalar);                                             //this = this./scalar   sse optimized
        float prod() const;                                                 //f = x0*x1*x2*...xN  sse optimized
        void inter2(const vec2D& src, vec2D& dst_grdx, vec2D& dst_grdy);    //biliniar 2d interpolation, grd temp buffers width = this->size height=2
        //zero offset array operations//////////////////////////////////////

        //mat operations
        float mconv(const float* a, const float* b, unsigned int size) const;       //a0*b0+a1*b1+a2*b2+ ... An*Bn  sse
        void conv2D(const vec2D& a, const vec2D& filter);                           //this = a*filter  convolution with filter
        void convolve(const vec2D& a, const vec2D& filter);                           //this = a*filter  convolution with filter
        void conv2D(const vec2D& a, const vec2D& re, const vec2D& im);              //this = convolution with complex filter [re i*im]

        void minmax(float& min, float& max) const;                                  //get min max values
        float maxval() const;                                                       //get max value
        float minval() const;                                                       //get min value
        void maxval(float& max, int& x, int& y,                                     //get max value with [x,y] coord
                    int sizex = 1, int sizey = 1, int dx = 0, int dy = 0) const;    //from offset dx,dy in rect [sizex,sizey]

        void normalize(float a, float b);                                           //normalize to a...b range  x1 = x-min
        void histeq(vec2D& hist);                                                   //histogram normalization vector in 0...255 range -> to 0...1.0 range
        void fliplr();                                                              //flip matrix along vertical axis


// Access
        inline int xfirst() const;                                                 //return first col index into array
        inline int yfirst() const;                                                 //return first row index into array
        inline int xlast() const;                                                  //return last col index into array
        inline int ylast() const;                                                  //return last row index into array
        inline unsigned int width() const;                                         //width of array
        inline unsigned int height() const;                                        //height size of array
        inline unsigned int length() const;                                        //total size of array width*height
        inline const float* data(int y, int x) const;


private:
        int m_xoffset, m_yoffset;
        int m_xlast, m_ylast;
        unsigned int m_width;
        unsigned int m_height;
        float** m_data;               //offseted data


        void init(unsigned int ysize, unsigned int xsize,
                  int yoffset = 0, int xoffset = 0);                           //allocate m_data only
        void init(const vec2D &v);                                              //fill this->m_data with v.m_data[:][:]
        void close(void);                                                       //deallocate m_data


};



/*
  arrange    -3 -2 -1 0 +1 +2  indeces array
          -3
          -2
          -1
           0          *
          +1
          +2

    vec1D v(6,6,-3,-3);
     v(-3,-3) = val1;
     v(-2,-3) = val2;
     ...
     v(2,2) = valN

*/


inline int vec2D::xfirst() const
{
        return m_xoffset;
}

inline int vec2D::yfirst() const
{
        return m_yoffset;
}

inline int vec2D::xlast() const
{
        return m_xlast;
}

inline int vec2D::ylast() const
{
        return m_ylast;
}

inline unsigned int vec2D::width() const
{
        return m_width;
}

inline unsigned int vec2D::height() const
{
        return m_height;
}

inline unsigned int vec2D::length() const
{
        return m_width*m_height;
}

inline const vec2D& vec2D::operator=(const float * pf)
{
        vec2D& v = *this;
        for (int y = v.yfirst(); y <= v.ylast(); y++)
                for (int x = v.xfirst(); x <= v.xlast(); x++)
                        v(y, x) = *pf++;

        return *this;
}

inline const vec2D& vec2D::operator=(const vec2D & v)
{
        if (this == &v) {
                return *this;
        } else if (m_width == v.width() && m_height == v.height()) {  //equal size arrays?
                if (m_xoffset != v.xfirst() || m_yoffset != v.yfirst()) {   //equal offsets? correct them if not
                        m_data += m_yoffset;
                        for (unsigned int j = 0; j < m_height; j++)
                                m_data[j] = m_data[j] + m_xoffset - v.xfirst();
                        m_data -= v.yfirst();

                        m_xoffset = v.xfirst();
                        m_yoffset = v.yfirst();
                }
                init(v);
        } else {  //create a complete copy from v
                close();
                init(v.height(), v.width(), v.yfirst(), v.xfirst());
                init(v);
        }
        return *this;
}

inline float& vec2D::operator()(int y, int x)
{
        return m_data[y][x];
}

inline float vec2D::operator()(int y, int x) const
{
        return m_data[y][x];
}

inline float& vec2D::operator[](unsigned int i)
{
        return m_data[i%m_height][i/m_height];     //m_height - MATLAB order operator v[:]
}

inline float vec2D::operator[](unsigned int i) const
{
        return m_data[i%m_height][i/m_height];
}

inline float vec2D::get(int y, int x) const
{
        if (x < xfirst()) x = xfirst() + (xfirst() - x);
        else if (x > xlast()) x = xlast() - (x - xlast());
        if (y < yfirst()) y = yfirst() + (yfirst() - y);
        else if (y > ylast()) y = ylast() - (y - ylast());

        return m_data[y][x];
}

inline const vec2D& vec2D::operator*(const vec2D& b) const     // c = a*b -> C = this*B
{
        const vec2D& a = *this;
        if (a.xfirst() != 0 || a.yfirst() != 0)
                return a;

        vec2D* pc = new vec2D(a.height(), b.width());             //0-offseted
        vec2D& c = *pc;

        for (unsigned int y = 0; y < c.height(); y++) {
                for (unsigned int x = 0; x < c.width(); x++) {
                        for (unsigned int i = 0; i < a.width(); i++)                     //a and b must be with zero offsets
                                c(y, x) += a(y, i) * b(i, x);
                }
        }

        return c;
}

inline const float* vec2D::data(int y, int x) const
{
        return m_data[y] + x;
}

inline float vec2D::mconv(const float* a, const float* b, unsigned int size) const
{
        float z = 0.0f, fres = 0.0f;
        __declspec(align(16)) float ftmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        __m128 mres;

        if ((size / 4) != 0) {
                mres = _mm_load_ss(&z);
                for (unsigned int i = 0; i < size / 4; i++)
                        mres = _mm_add_ps(mres, _mm_mul_ps(_mm_load_ps(&a[4*i]), _mm_load_ps(&b[4*i])));

                //mres = a,b,c,d
                __m128 mv1 = _mm_movelh_ps(mres, mres);     //a,b,a,b
                __m128 mv2 = _mm_movehl_ps(mres, mres);     //c,d,c,d
                mres = _mm_add_ps(mv1, mv2);                //res[0],res[1]

                _mm_store_ps(ftmp, mres);

                fres = ftmp[0] + ftmp[1];
        }

        if ((size % 4) != 0) {
                for (unsigned int i = size - size % 4; i < size; i++)
                        fres += a[i] * b[i];
        }

        return fres;
}


#endif
