double mu, sigma;
switch(get_cpuid()) {
    case 2: // node 1
        mu    = 2.505687e-06 + 7.394729e-11*mnk + -2.445496e-10*mn + 4.252655e-10*mk + 2.239360e-09*nk;
        sigma = 3.426729e-07 + 1.635744e-13*mnk + 6.613422e-13*mn + 5.707175e-11*mk + 5.838398e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 3: // node 1
        mu    = 2.480229e-06 + 7.000482e-11*mnk + -2.785525e-10*mn + 3.394621e-10*mk + 2.001561e-09*nk;
        sigma = 4.284178e-07 + 3.229869e-13*mnk + -2.000178e-11*mn + 5.708675e-11*mk + 4.273472e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 4: // node 2
        mu    = 2.585375e-06 + 7.299057e-11*mnk + -2.658263e-10*mn + 5.351639e-10*mk + 2.331181e-09*nk;
        sigma = 4.405208e-07 + 2.516536e-13*mnk + -9.906506e-12*mn + 7.764381e-11*mk + 7.425394e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 5: // node 2
        mu    = 2.691188e-06 + 6.941351e-11*mnk + -2.612029e-10*mn + 4.776279e-10*mk + 2.098164e-09*nk;
        sigma = 5.304273e-07 + 2.493957e-13*mnk + -1.006722e-11*mn + 4.094791e-11*mk + 3.874917e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 6: // node 3
        mu    = 2.512229e-06 + 7.140143e-11*mnk + -2.004919e-10*mn + 8.924510e-10*mk + 2.278175e-09*nk;
        sigma = 3.701914e-07 + 1.479307e-13*mnk + 3.246915e-12*mn + 2.372402e-11*mk + 1.127836e-10*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 7: // node 3
        mu    = 2.591000e-06 + 6.858651e-11*mnk + -2.150188e-10*mn + 5.003876e-10*mk + 2.166557e-09*nk;
        sigma = 5.681205e-07 + 1.542616e-13*mnk + -4.363612e-12*mn + 4.794356e-11*mk + 5.466080e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 8: // node 4
        mu    = 2.509771e-06 + 6.904071e-11*mnk + -2.360159e-10*mn + 3.736541e-10*mk + 2.134810e-09*nk;
        sigma = 3.273841e-07 + 2.666216e-13*mnk + -8.993575e-12*mn + 2.140807e-11*mk + 3.970023e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 9: // node 4
        mu    = 2.607208e-06 + 6.985533e-11*mnk + -2.764269e-10*mn + 4.109585e-10*mk + 2.026962e-09*nk;
        sigma = 4.550966e-07 + 2.812821e-13*mnk + -1.468970e-11*mn + 6.237992e-11*mk + 4.246233e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 10: // node 5
        mu    = 2.613646e-06 + 7.269522e-11*mnk + -2.132326e-10*mn + 7.772362e-10*mk + 2.193113e-09*nk;
        sigma = 3.621765e-07 + 1.145244e-13*mnk + 1.179463e-11*mn + 2.334804e-11*mk + 6.581988e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 11: // node 5
        mu    = 2.585312e-06 + 6.606024e-11*mnk + -1.545941e-10*mn + 8.976167e-10*mk + 2.598591e-09*nk;
        sigma = 5.915091e-07 + 3.510266e-13*mnk + -2.814343e-11*mn + 9.988660e-12*mk + 4.654929e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 12: // node 6
        mu    = 2.647667e-06 + 7.391871e-11*mnk + -2.323033e-10*mn + 6.068064e-10*mk + 2.175669e-09*nk;
        sigma = 4.287263e-07 + 1.625383e-13*mnk + 5.136388e-12*mn + 4.325477e-11*mk + 4.450112e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 13: // node 6
        mu    = 2.388167e-06 + 6.433067e-11*mnk + -9.948719e-11*mn + 1.144963e-09*mk + 2.845869e-09*nk;
        sigma = 4.179725e-07 + 3.132253e-13*mnk + -2.270656e-11*mn + 2.699975e-11*mk + 3.615856e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 14: // node 7
        mu    = 2.477583e-06 + 6.995681e-11*mnk + -2.459865e-10*mn + 7.136085e-10*mk + 2.094251e-09*nk;
        sigma = 3.413847e-07 + 1.476828e-13*mnk + 2.662202e-13*mn + 4.436943e-11*mk + 8.369002e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 15: // node 7
        mu    = 2.469771e-06 + 6.361542e-11*mnk + -6.556833e-11*mn + 1.300629e-09*mk + 2.935818e-09*nk;
        sigma = 5.523769e-07 + 2.928643e-13*mnk + -1.095698e-11*mn + 1.347378e-11*mk + 1.196860e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 16: // node 8
        mu    = 2.426479e-06 + 7.306837e-11*mnk + -2.284432e-10*mn + 6.106052e-10*mk + 1.996301e-09*nk;
        sigma = 3.844988e-07 + 2.581578e-13*mnk + 2.864728e-12*mn + 4.726106e-13*mk + 5.724820e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 17: // node 8
        mu    = 2.521667e-06 + 6.773985e-11*mnk + -1.840563e-10*mn + 6.007267e-10*mk + 2.268158e-09*nk;
        sigma = 4.725194e-07 + 2.478775e-13*mnk + 1.026035e-12*mn + -2.663242e-12*mk + 1.059490e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 18: // node 9
        mu    = 2.741833e-06 + 6.603880e-11*mnk + -1.654358e-10*mn + 9.223434e-10*mk + 2.665966e-09*nk;
        sigma = 3.496762e-07 + 4.767849e-13*mnk + -3.876633e-11*mn + 1.034907e-11*mk + 5.192573e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 19: // node 9
        mu    = 2.603000e-06 + 6.663587e-11*mnk + -1.746079e-10*mn + 8.064748e-10*mk + 2.527964e-09*nk;
        sigma = 4.509433e-07 + 3.045085e-13*mnk + -2.037512e-11*mn + 2.816047e-11*mk + 5.765530e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 20: // node 10
        mu    = 2.446396e-06 + 6.719571e-11*mnk + -1.529947e-10*mn + 6.673197e-10*mk + 2.417459e-09*nk;
        sigma = 3.877258e-07 + 2.002332e-13*mnk + -8.821556e-12*mn + 5.841179e-11*mk + 5.071971e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 21: // node 10
        mu    = 2.625729e-06 + 6.664561e-11*mnk + -1.607611e-10*mn + 7.749566e-10*mk + 2.475782e-09*nk;
        sigma = 4.006853e-07 + 2.385598e-13*mnk + -1.093045e-11*mn + 1.914530e-11*mk + 3.817311e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 22: // node 11
        mu    = 2.535250e-06 + 6.614710e-11*mnk + -1.562363e-10*mn + 8.455955e-10*mk + 2.623094e-09*nk;
        sigma = 4.155710e-07 + 4.022548e-13*mnk + -3.232944e-11*mn + 1.175355e-11*mk + 4.932645e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 23: // node 11
        mu    = 2.595083e-06 + 7.117132e-11*mnk + -2.294363e-10*mn + 8.601404e-10*mk + 1.920661e-09*nk;
        sigma = 4.615134e-07 + 2.390826e-13*mnk + 1.379259e-12*mn + 3.340658e-11*mk + 6.566121e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 24: // node 12
        mu    = 2.654500e-06 + 6.697900e-11*mnk + -1.525997e-10*mn + 6.789790e-10*mk + 2.404847e-09*nk;
        sigma = 4.232980e-07 + 2.361973e-13*mnk + -1.318395e-11*mn + 2.830758e-11*mk + 4.751448e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 25: // node 12
        mu    = 2.695771e-06 + 6.613499e-11*mnk + -1.589997e-10*mn + 8.498932e-10*mk + 2.600466e-09*nk;
        sigma = 4.287671e-07 + 3.223844e-13*mnk + -1.371638e-11*mn + 1.538294e-11*mk + 1.414765e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 26: // node 13
        mu    = 2.637208e-06 + 7.006534e-11*mnk + -2.549513e-10*mn + 7.921601e-10*mk + 2.139585e-09*nk;
        sigma = 4.060366e-07 + 1.678107e-13*mnk + 2.131890e-12*mn + 1.084745e-10*mk + 6.971995e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 27: // node 13
        mu    = 2.706479e-06 + 6.732821e-11*mnk + -2.065264e-10*mn + 6.823099e-10*mk + 2.437552e-09*nk;
        sigma = 5.049250e-07 + 1.863073e-13*mnk + -5.574299e-12*mn + 3.314776e-11*mk + 5.480737e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 28: // node 14
        mu    = 2.690521e-06 + 7.041220e-11*mnk + -2.670919e-10*mn + 6.049498e-10*mk + 2.089691e-09*nk;
        sigma = 4.460643e-07 + 1.561241e-13*mnk + 4.331978e-12*mn + 1.180545e-10*mk + 6.676003e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 29: // node 14
        mu    = 2.606646e-06 + 6.879902e-11*mnk + -2.313754e-10*mn + 3.629877e-10*mk + 2.194596e-09*nk;
        sigma = 4.765937e-07 + 1.939126e-13*mnk + -7.063286e-12*mn + 1.110161e-11*mk + 6.083831e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 30: // node 15
        mu    = 2.774729e-06 + 7.360603e-11*mnk + -1.951662e-10*mn + 6.284668e-10*mk + 2.251370e-09*nk;
        sigma = 5.048611e-07 + 1.836693e-13*mnk + 6.102690e-12*mn + 3.622790e-11*mk + 4.296437e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 31: // node 15
        mu    = 2.561438e-06 + 6.710256e-11*mnk + -1.992883e-10*mn + 7.234953e-10*mk + 2.431414e-09*nk;
        sigma = 4.787896e-07 + 3.219301e-13*mnk + -2.112881e-11*mn + 1.724434e-11*mk + 3.794455e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 32: // node 16
        mu    = 2.881562e-06 + 7.102840e-11*mnk + -2.089913e-10*mn + 1.042113e-09*mk + 2.218115e-09*nk;
        sigma = 3.799975e-07 + 4.640392e-13*mnk + -1.689604e-11*mn + -7.141750e-12*mk + 9.012388e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 33: // node 16
        mu    = 2.698812e-06 + 6.826538e-11*mnk + -2.317713e-10*mn + 5.544147e-10*mk + 2.329573e-09*nk;
        sigma = 4.796515e-07 + 1.865540e-13*mnk + -7.839847e-12*mn + 4.115532e-11*mk + 6.216831e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 34: // node 17
        mu    = 2.657458e-06 + 7.246203e-11*mnk + -2.308027e-10*mn + 5.378567e-10*mk + 2.453706e-09*nk;
        sigma = 3.380224e-07 + 1.675055e-13*mnk + -1.497576e-12*mn + 5.699775e-11*mk + 7.053454e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 35: // node 17
        mu    = 2.539813e-06 + 6.963251e-11*mnk + -2.660416e-10*mn + 3.589212e-10*mk + 2.061668e-09*nk;
        sigma = 3.696011e-07 + 3.543268e-13*mnk + -2.681616e-11*mn + 3.556817e-11*mk + 4.771028e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 36: // node 18
        mu    = 2.546271e-06 + 7.188030e-11*mnk + -1.889568e-10*mn + 9.097041e-10*mk + 2.387038e-09*nk;
        sigma = 3.561689e-07 + 6.826456e-13*mnk + -6.040792e-11*mn + 2.908558e-11*mk + 5.766961e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 37: // node 18
        mu    = 2.304625e-06 + 6.561569e-11*mnk + -1.404717e-10*mn + 9.544250e-10*mk + 2.624172e-09*nk;
        sigma = 4.871249e-07 + 4.316466e-13*mnk + -3.511331e-11*mn + 9.556236e-12*mk + 3.879602e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 38: // node 19
        mu    = 2.511979e-06 + 6.889241e-11*mnk + -2.055221e-10*mn + 5.922121e-10*mk + 2.141948e-09*nk;
        sigma = 4.577048e-07 + 1.460528e-13*mnk + 5.652398e-12*mn + 4.614212e-11*mk + 4.601130e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 39: // node 19
        mu    = 2.836625e-06 + 6.864001e-11*mnk + -2.373024e-10*mn + 4.537737e-10*mk + 2.225277e-09*nk;
        sigma = 4.441739e-07 + 2.466384e-13*mnk + -9.344609e-12*mn + 2.104507e-11*mk + 3.973981e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 40: // node 20
        mu    = 2.491875e-06 + 7.044862e-11*mnk + -2.260625e-10*mn + 7.166291e-10*mk + 1.984063e-09*nk;
        sigma = 3.693793e-07 + 2.022760e-13*mnk + -5.095333e-12*mn + 1.095160e-10*mk + 7.306089e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 41: // node 20
        mu    = 2.722417e-06 + 6.696748e-11*mnk + -2.063029e-10*mn + 7.473446e-10*mk + 2.567997e-09*nk;
        sigma = 5.249457e-07 + 1.784877e-13*mnk + -5.069879e-12*mn + 1.236338e-11*mk + 5.602746e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 42: // node 21
        mu    = 2.638604e-06 + 7.122294e-11*mnk + -2.304068e-10*mn + 9.244350e-10*mk + 1.781784e-09*nk;
        sigma = 4.087682e-07 + 2.833467e-13*mnk + -1.096634e-11*mn + 2.522037e-11*mk + 5.956344e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 43: // node 21
        mu    = 2.458063e-06 + 6.952317e-11*mnk + -2.503400e-10*mn + 4.412917e-10*mk + 2.039829e-09*nk;
        sigma = 4.711367e-07 + 2.338999e-13*mnk + -1.027532e-11*mn + 6.567303e-11*mk + 5.018173e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 44: // node 22
        mu    = 2.452833e-06 + 6.879576e-11*mnk + -2.407274e-10*mn + 4.426200e-10*mk + 2.164902e-09*nk;
        sigma = 3.742780e-07 + 2.215716e-13*mnk + -1.252130e-11*mn + 4.261556e-11*mk + 4.472830e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 45: // node 22
        mu    = 2.563396e-06 + 6.941590e-11*mnk + -2.632147e-10*mn + 4.048982e-10*mk + 2.137393e-09*nk;
        sigma = 4.302435e-07 + 3.592876e-13*mnk + -2.564776e-11*mn + 4.320514e-11*mk + 4.932294e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 46: // node 23
        mu    = 2.529104e-06 + 6.996612e-11*mnk + -2.576589e-10*mn + 5.956454e-10*mk + 1.934548e-09*nk;
        sigma = 3.312401e-07 + 2.803529e-13*mnk + -5.331287e-12*mn + 7.733383e-11*mk + 2.268368e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 47: // node 23
        mu    = 2.721125e-06 + 6.575375e-11*mnk + -1.250152e-10*mn + 8.399464e-10*mk + 2.592213e-09*nk;
        sigma = 5.748203e-07 + 1.261773e-13*mnk + 5.029305e-13*mn + 1.307661e-11*mk + 4.893509e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 48: // node 24
        mu    = 2.407354e-06 + 7.010583e-11*mnk + -2.459946e-10*mn + 4.334577e-10*mk + 1.852938e-09*nk;
        sigma = 3.020854e-07 + 2.313108e-13*mnk + -7.082031e-12*mn + 6.169913e-11*mk + 3.971285e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 49: // node 24
        mu    = 2.419646e-06 + 6.754723e-11*mnk + -2.002099e-10*mn + 5.561133e-10*mk + 2.338643e-09*nk;
        sigma = 5.037270e-07 + 2.196482e-13*mnk + -1.962580e-12*mn + -9.796227e-12*mk + 1.798865e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 50: // node 25
        mu    = 2.448167e-06 + 7.783130e-11*mnk + -1.971879e-10*mn + 5.547125e-10*mk + 1.576901e-09*nk;
        sigma = 3.506372e-07 + 6.836720e-13*mnk + -5.915784e-11*mn + 3.829178e-11*mk + 7.802097e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 51: // node 25
        mu    = 2.380917e-06 + 6.394314e-11*mnk + -1.160058e-10*mn + 1.299952e-09*mk + 2.955523e-09*nk;
        sigma = 4.676722e-07 + 3.618144e-13*mnk + -2.999729e-11*mn + 2.415284e-11*mk + 4.073623e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 52: // node 26
        mu    = 2.331292e-06 + 6.888705e-11*mnk + -1.997542e-10*mn + 3.701790e-10*mk + 2.121579e-09*nk;
        sigma = 2.725037e-07 + 2.688719e-13*mnk + -7.597633e-12*mn + 2.900857e-11*mk + 3.597657e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 53: // node 26
        mu    = 2.708312e-06 + 7.127367e-11*mnk + -3.123161e-10*mn + 3.314873e-10*mk + 1.991597e-09*nk;
        sigma = 6.220409e-07 + 4.055665e-13*mnk + -3.320611e-11*mn + 5.901314e-11*mk + 8.107703e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 54: // node 27
        mu    = 2.801604e-06 + 7.053132e-11*mnk + -3.275745e-10*mn + 3.602435e-10*mk + 2.057926e-09*nk;
        sigma = 3.755145e-07 + 2.666626e-13*mnk + -1.626989e-11*mn + 4.422960e-11*mk + 6.492428e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 55: // node 27
        mu    = 2.750687e-06 + 7.018566e-11*mnk + -2.994676e-10*mn + 4.031788e-10*mk + 2.085565e-09*nk;
        sigma = 4.612120e-07 + 4.489872e-13*mnk + -3.344311e-11*mn + 6.659179e-11*mk + 5.084358e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 56: // node 28
        mu    = 2.766792e-06 + 7.402421e-11*mnk + -2.684769e-10*mn + 4.601428e-10*mk + 2.041225e-09*nk;
        sigma = 3.861498e-07 + 2.366380e-13*mnk + -2.817787e-12*mn + 1.373686e-11*mk + 5.895634e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 57: // node 28
        mu    = 2.424313e-06 + 6.697333e-11*mnk + -1.623358e-10*mn + 6.912950e-10*mk + 2.409605e-09*nk;
        sigma = 4.114536e-07 + 2.117366e-13*mnk + -1.424349e-11*mn + 3.578460e-11*mk + 5.340794e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 58: // node 29
        mu    = 2.374875e-06 + 6.931938e-11*mnk + -2.479668e-10*mn + 4.881747e-10*mk + 2.088861e-09*nk;
        sigma = 3.129785e-07 + 2.146303e-13*mnk + -6.821791e-12*mn + 5.798791e-11*mk + 5.168179e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 59: // node 29
        mu    = 2.542750e-06 + 6.811881e-11*mnk + -2.054642e-10*mn + 4.752504e-10*mk + 2.256994e-09*nk;
        sigma = 5.641988e-07 + 2.812581e-13*mnk + -1.985059e-11*mn + 2.010326e-11*mk + 4.612689e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 62: // node 31
        mu    = 2.458521e-06 + 6.698166e-11*mnk + -1.628124e-10*mn + 7.055535e-10*mk + 2.388120e-09*nk;
        sigma = 3.383245e-07 + 2.239010e-13*mnk + -1.007746e-11*mn + 1.604762e-11*mk + 4.022998e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 63: // node 31
        mu    = 2.456604e-06 + 6.878254e-11*mnk + -2.057684e-10*mn + 5.236678e-10*mk + 2.108636e-09*nk;
        sigma = 3.696952e-07 + 2.933489e-13*mnk + -1.389034e-11*mn + 3.604661e-11*mk + 3.203429e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 64: // node 32
        mu    = 2.540292e-06 + 6.633782e-11*mnk + -1.676669e-10*mn + 8.347438e-10*mk + 2.569045e-09*nk;
        sigma = 3.622983e-07 + 2.924033e-13*mnk + -1.750039e-11*mn + 9.045783e-12*mk + 4.445465e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    case 65: // node 32
        mu    = 2.729750e-06 + 6.629256e-11*mnk + -1.700038e-10*mn + 8.741856e-10*mk + 2.547823e-09*nk;
        sigma = 4.838515e-07 + 1.787325e-13*mnk + -4.349947e-12*mn + 2.925361e-11*mk + 4.626528e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
    default:
        mu    = 2.573935e-06 + 6.913763e-11*mnk + -2.113842e-10*mn + 6.562226e-10*mk + 2.267745e-09*nk;
        sigma = 4.279201e-07 + 2.699896e-13*mnk + -1.238246e-11*mn + 3.595527e-11*mk + 5.109858e-11*nk;
        return mu + random_halfnormal_shifted(0, sigma);
}
