#include <iostream>
#include <bitset>
#include <math.h>
#include "PytorchBridge.hpp"

#include<iostream>
#include<fstream>

using namespace std;
namespace flopoco {
    void debug_print(int index, int64_t tmplx, int64_t tmplw, int64_t lx, int64_t lw, int64_t lwx, double lwxFloat,
                     double wxFloat, int64_t summand, int64_t acc) {
            cout << index << "th element:" << endl <<
             "received lw: " << bitset<10>(tmplw) << endl <<
             "received lx: " << bitset<10>(tmplx) << endl <<
             "lw: " << bitset<10>(lw) << " or " << lw << endl <<
             "lx: " << bitset<10>(lx) << endl <<
             "lwx: " << bitset<10>(lwx) << " or " << double(lwx) << endl <<
             "lwxFloat: " << lwxFloat << endl <<
             "wxFloat: " << wxFloat << endl <<
             "summand: " << bitset<10>(summand) <<
             ", acc: " << bitset<10>(acc) << endl ; 
    }

    /**
     * 
     * @param lxs The array of input values.
     * @param lws The array of weight values.
     * @param bias The bias value.
     * @param msb The most significant bit.
     * @param lsb The least significant bit.
     * @param prevLayerWidth The width of the previous layer.
     * @param expLsb The least significant bit of the exponent.
     * @param isLastLayer Indicates whether this is the last layer.
     * @return The result of the emulation.
     * 
     *     // lx =   xxxx xxxx   -log(x) in [0, 2^(msb+1) - 2^lsb] => x in ]0, 1]
     *     // lw = S xxxx xxxx   -log(w) in [0, 2^(msb+1) - 2^lsb] => w in [-1, 1[ without 0
     */
    int64_t do_emulate(int64_t lxs[], int64_t lws[], int64_t bias, int msb, int lsb, int prevLayerWidth,
                         int expLsb, bool isLastLayer) {
        int64_t tmplx, tmplw, valFilter, filtered_lw, lx, lw, signlw, og_lwx, lwx, summand, acc, a;
        double wxFloat, accFloat, loga;
        int inputLogSize = msb - lsb + 1;
        valFilter = (1ll << inputLogSize) - 1;
        acc = bias; // Virtually useless, we could accumulate in b

        for (int i = 0; i < prevLayerWidth; i++) {
            tmplx = lxs[i];
            tmplw = lws[i];

            bool negw = tmplw >> inputLogSize;
            lw = tmplw & valFilter; // unsigned, -log(w) // add abs 
            lx = tmplx; // unsigned, -log(x)
            lwx = lw + lx; // -log(wx) // hier passiert die Multiplikation als Addition
            // lwx reprsents fixpoint numbers ("ohne Komma")

            double lwxFloat = -double(lwx) * exp2(lsb); // exact, log(wx), alignment
            wxFloat = exp2(lwxFloat); // rounding @lsb=-53, 2**x // hier wird zurueck gerechnet Produkt ins lineare
            // hier print wxFloat
            // muesste gleiche wie product sein
            wxFloat *= exp2(-expLsb); // exact, alignment // expLsb == sumLSB
            summand = int64_t(round(wxFloat)); // rounding @expLsb TODO: round or truncate ?

            //cout << "### WXFLOAT in CPP: " << round(wxFloat) * exp2(expLsb) << endl;

            // summand = double (summand) * exp2(expLsb); // muesste quantized proudct sein
            // hier auch print summand aber in double gecastet  * 2 ** explsb
            summand = summand & ((1ll << (2 - expLsb)) - 1); // Making sure summand is small

            if(negw)
                acc -= summand;
            else
                acc += summand;
            //debug_print(i, tmplx, tmplw, lx, lw, lwx, lwxFloat, wxFloat, summand, acc);
        }

        // ofstream file;
        // file.open("activations.csv", ios::app);
        // file << double(acc) / (1ll << -expLsb) << ", ";
        // file.close();
        // if(abs(double(acc) * exp2(expLsb)) > 5)
        //     cout << "Bang " << double(acc) * exp2(expLsb) << endl;

        // Clipped RELU activation function
        if(isLastLayer == false) {

            cout << "Acc before : " << acc  << endl;

            if (acc > (1ll << -expLsb)) // High cut
                acc = 1ll << -expLsb;

            if(acc <= 0) // Low cut
                a = valFilter; // Smallest representable value
            else { // Translate back to log domain
                accFloat = double(acc) * exp2(expLsb); // exact

                cout << "AccFloat : " << accFloat  << endl;

                loga = log2(accFloat); // rounding @lsb=-53
                loga *= exp2(-lsb);

                cout << "loga : " << loga  << endl;

                a = int64_t(round(-loga)); // >0, unsigned, rounding @lsb

                cout << "a : " << exp2(-double(a) * exp2(lsb))  << endl;

                if(a > valFilter) {
                    cout << "Overflow ! Diff is: " << -a - valFilter << endl << endl;
                    a = valFilter;
                }
            }
        } else {
            if(acc <= 0) // Low cut
                a = 0; // Smallest respresentable value
            else { // Final layer doesn't need to restrict bit width nor tranlation to log domain
                a = acc;
            }
        }

        cout << "return a : " << a  << endl;

        return a;
    }
}
