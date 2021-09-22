#pragma once
#include <tuple>
#include <utility> 
#include <variant>


template<std::size_t I = 0, typename FuncT, typename... Tp>
void for_each(std::tuple<Tp...>& t, FuncT f)
{ 
	f(I, std::get<I>(t));
	if constexpr (I+1 < sizeof...(Tp))
		for_each<I + 1, FuncT, Tp...>(t, f);
}

struct DynamicHCHCW
{
    int NHo;
    int NCo;
    int NHi;
    int NW;
    int NCi;
    int NCu;
    int NHu;
    int NWu;
    DynamicHCHCW(int Ho, int Co, int Hi, int Ci, int W, int Cu, int Hu, int Wu) 
        : NHo(Ho), NCo(Co), NHi(Hi), NCi(Ci), NW(W), NCu(Cu), NHu(Hu), NWu(Wu)
    {
    }
    template <typename F>
    void for_each_hcw(F f) {
        for(int ho = 0; ho < NHo && ho*NHi < NHu; ho++)
        for(int hi = 0; hi < NHi; hi++)
        {
            int h = ho*NHi + hi;
            if (h >= NHu)
                break;
            for(int co = 0; co < NCo && co*NCi < NCu; co++)
            for(int ci = 0; ci < NCi; ci++) 
            {
                int c = co*NCi + ci;
                if (c >= NCu)
                    break;
                for(int w = 0; w < NW && w < NWu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
    }
    template <typename F>
    void for_each_chw(F f) {
        for(int co = 0; co < NCo && co*NCi < NCu; co++)
        for(int ci = 0; ci < NCi; ci++) 
        {
            int c = co*NCi + ci;
            if (c >= NCu)
                break;
            for(int ho = 0; ho < NHo && ho*NHi < NHu; ho++)
            for(int hi = 0; hi < NHi; hi++)
            {
                int h = ho*NHi + hi;
                if (h >= NHu)
                    break;
                for(int w = 0; w < NW && w < NWu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
    }
    constexpr int size() const { return NCu*NHu*NWu; }
    constexpr int lowered_size() const { return NHo*NCo*NHi*NW*NCi; }
    int index(int co, int ci, int ho, int hi, int w) {
        return (((ho*NCo+co)*NHi+hi)*NCi+ci)*NW+w;
    }
    int index(int c, int h, int w) {
        int co = c / NCi;
        int ci = c - co*NCi;
        int ho = h / NHi;
        int hi = h - ho*NHi;
        return index(co, ci, ho, hi, w);
    }
    template <typename F>
    void for_each(F f) {
        for(int ho = 0; ho < NHo && ho * NHi < NHu; ho++)
        for(int co = 0; co < NCo && co * NCi < NCu; co++)
        for(int hi = 0; hi < NHi; hi++)
        {
            int h = ho*NHi + hi;
            if (h >= NHu)
                break;
            for(int ci = 0; ci < NCi; ci++) 
            {
                int c = co*NCi + ci;
                if (c >= NCu)
                    break;
                for(int w = 0; w < NW && w < NWu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
    }
};

struct DynamicHCHWC
{
    int NHo;
    int NCo;
    int NHi;
    int NW;
    int NCi;
    int NCu;
    int NHu;
    int NWu;
    DynamicHCHWC(int Ho, int Co, int Hi, int W, int Ci, int Cu, int Hu, int Wu) 
        : NHo(Ho), NCo(Co), NHi(Hi), NW(W), NCi(Ci), NCu(Cu), NHu(Hu), NWu(Wu)
    {
    }
    constexpr int size() const { return NCu*NHu*NWu; }
    constexpr int lowered_size() const { return NHo*NCo*NHi*NW*NCi; }
    int index(int co, int ci, int ho, int hi, int w) {
        return (((ho*NCo+co)*NHi+hi)*NW+w)*NCi+ci;
    }
    int index(int c, int h, int w) {
        int step = (NCu+NCo-1) / NCo;
        int co = c / step;
        int ci = c % step;
        int ho = h / NHi;
        int hi = h % NHi;
        return index(co, ci, ho, hi, w);
    }
    template <typename F>
    void for_each_hcw(F f) {
        int step = (NCu+NCo-1) / NCo;
        for(int ho = 0; ho < NHo && ho*NHi < NHu; ho++)
        for(int hi = 0; hi < NHi; hi++)
        {
            int h = ho*NHi + hi;
            if (h >= NHu)
                break;
            for(int co = 0; co < NCo && co*step < NCu; co++)
            for(int ci = 0; ci < NCi; ci++) 
            {
                int c = co*step + ci;
                if (c >= NCu)
                    break;
                for(int w = 0; w < NW && w < NWu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
    }
    template <typename F>
    void for_each_chw(F f) {
        int step = (NCu+NCo-1) / NCo;
        for(int co = 0; co < NCo && co*step < NCu; co++)
        for(int ci = 0; ci < NCi; ci++) 
        {
            int c = co*step + ci;
            if (c >= NCu)
                break;
            for(int ho = 0; ho < NHo && ho*NHi < NHu; ho++)
            for(int hi = 0; hi < NHi; hi++)
            {
                int h = ho*NHi + hi;
                if (h >= NHu)
                    break;
                for(int w = 0; w < NW && w < NWu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
    }
    template <typename F>
    void for_each(F f) {
        int step = (NCu+NCo-1) / NCo;
        for(int ho = 0; ho < NHo && ho * NHi < NHu; ho++)
        for(int co = 0; co < NCo && co * step < NCu; co++)
        for(int hi = 0; hi < NHi; hi++)
        {
            int h = ho*NHi + hi;
            if (h >= NHu)
                break;
            for(int w = 0; w < NW && w < NWu; w++) 
            for(int ci = 0; ci < NCi; ci++) 
            {
                int c = co*step + ci;
                if (c >= NCu)
                    break;
                f(c, co, ci, h, ho, hi, w);
            }
        }
    }
};


using LoweringInfo = std::variant<DynamicHCHWC, DynamicHCHCW>;

template <int Ho, int Co, int Hi, int Ci, int W, int Cu, int Hu, int Wu>
struct HCHCW
{
    static const int NHu = Hu;
    static const int NCu = Cu;
    static const int NWu = Wu;
    static const int NHo = Ho;
    static const int NCo = Co;
    static const int NHi = Hi;
    static const int NCi = Ci;
    static const int NW = W;
    constexpr int size() const { return Cu*Hu*Wu; }
    constexpr int lowered_size() const { return Ho*Co*Hi*W*Ci; }
    int index(int co, int ci, int ho, int hi, int w) {
        return (((ho*Co+co)*Hi+hi)*Ci+ci)*W+w;
    }
    int index(int c, int h, int w) {
        int co = c / Ci;
        int ci = c - co*Ci;
        int ho = h / Hi;
        int hi = h - ho*Hi;
        return index(co, ci, ho, hi, w);
    }
    template <typename F>
    void for_each_chw(F f) {
        for(int c = 0; c < Cu; c++) {
            int co = c / Ci;
            int ci = c % Ci;
            for(int h = 0; h < Hu; h++) {
                int ho = h / Hi;
                int hi = h % Hi;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
        /*for(int co = 0; co < Co && co*Ci < Cu; co++)
        for(int ci = 0; ci < Ci; ci++) 
        {
            int c = co*Ci + ci;
            if (c >= Cu)
                break;
            for(int ho = 0; ho < Ho && ho*Hi < Hu; ho++)
            for(int hi = 0; hi < Hi; hi++)
            {
                int h = ho*Hi + hi;
                if (h >= Hu)
                    break;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }*/
    }
    template <typename F>
    void for_each_hcw(F f) {
        for(int h = 0; h < Hu; h++) {
            int ho = h / Hi;
            int hi = h % Hi;
            for(int c = 0; c < Cu; c++) {
                int co = c / Ci;
                int ci = c % Ci;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
        /*
        for(int ho = 0; ho < Ho && ho*Hi < Hu; ho++)
        for(int hi = 0; hi < Hi; hi++)
        {
            int h = ho*Hi + hi;
            if (h >= Hu)
                break;
            for(int co = 0; co < Co && co*Ci < Cu; co++)
            for(int ci = 0; ci < Ci; ci++) 
            {
                int c = co*Ci + ci;
                if (c >= Cu)
                    break;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
        */
    }
    template <typename F>
    void for_each(F f) {
        for(int ho = 0; ho < Ho && ho * Hi < Hu; ho++)
        for(int co = 0; co < Co && co * Ci < Cu; co++)
        for(int hi = 0; hi < Hi; hi++)
        {
            int h = ho*Hi + hi;
            if (h >= Hu)
                break;
            for(int ci = 0; ci < Ci; ci++) 
            {
                int c = co*Ci + ci;
                if (c >= Cu)
                    break;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
    }
};

template <int Ho, int Co, int Hi, int W, int Ci, int Cu, int Hu, int Wu>
struct HCHWC 
{
    static const int NHu = Hu;
    static const int NCu = Cu;
    static const int NWu = Wu;
    static const int NHo = Ho;
    static const int NCo = Co;
    static const int NHi = Hi;
    static const int NCi = Ci;
    static const int NW = W;
    constexpr int size() const { return Cu*Hu*Wu; }
    constexpr int lowered_size() const { return Ho*Co*Hi*W*Ci; }
    int index(int co, int ci, int ho, int hi, int w) {
        return (((ho*Co+co)*Hi+hi)*W+w)*Ci+ci;
    }
    int index(int c, int h, int w) {
        int step = (Cu+Co-1) / Co;
        int co = c / step;
        int ci = c % step;
        int ho = h / Hi;
        int hi = h % Hi;
        return index(co, ci, ho, hi, w);
    }
    template <typename F>
    void for_each_chw(F f) {
        int step = (Cu+Co-1) / Co;
        for(int c = 0; c < Cu; c++) {
            int co = c / step;
            int ci = c % step;
            for(int h = 0; h < Hu; h++) {
                int ho = h / Hi;
                int hi = h % Hi;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
        /*
        for(int co = 0; co < Co && co*Ci < Cu; co++)
        for(int ci = 0; ci < step; ci++) 
        {
            int c = co*step + ci;
            if (c >= Cu)
                break;
            for(int ho = 0; ho < Ho && ho*Hi < Hu; ho++)
            for(int hi = 0; hi < Hi; hi++)
            {
                int h = ho*Hi + hi;
                if (h >= Hu)
                    break;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
        */
    }
    template <typename F>
    void for_each_hcw(F f) {
        int step = (Cu+Co-1) / Co;
        for(int h = 0; h < Hu; h++) {
            int ho = h / Hi;
            int hi = h % Hi;
            for(int c = 0; c < Cu; c++) {
                int co = c / step;
                int ci = c % step;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
        /*
        for(int ho = 0; ho < Ho && ho*Hi < Hu; ho++)
        for(int hi = 0; hi < Hi; hi++)
        {
            int h = ho*Hi + hi;
            if (h >= Hu)
                break;
            for(int co = 0; co < Co && co*Ci < Cu; co++)
            for(int ci = 0; ci < Ci; ci++) 
            {
                int c = co*Ci + ci;
                if (c >= Cu)
                    break;
                for(int w = 0; w < W && w < Wu; w++) {
                    f(c, co, ci, h, ho, hi, w);
                }
            }
        }
        */
    }
    template <typename F>
    void for_each(F f) {
        int step = (Cu+Co-1) / Co;
        for(int ho = 0; ho < Ho && ho * Hi < Hu; ho++)
        for(int co = 0; co < Co && co * step < Cu; co++)
        for(int hi = 0; hi < Hi; hi++)
        {
            int h = ho*Hi + hi;
            if (h >= Hu)
                break;
            for(int w = 0; w < W && w < Wu; w++) 
            for(int ci = 0; ci < Ci; ci++) 
            {
                int c = co*step + ci;
                if (c >= Cu)
                    break;
                f(c, co, ci, h, ho, hi, w);
            }
        }
    }
};
