from math import log2


# This is a SIMD-optimized beta constant calculation using vectorized polynomial evaluation.
# It is used to calculate the beta constant for a given number of registers.
# The beta constant is used to calculate the beta distribution.
# The beta distribution is a probability distribution that is used to model the probability of a random variable taking on a value between 0 and 1.
# The beta distribution is defined by two parameters, alpha and beta.
# The alpha parameter is the shape parameter of the distribution.
# The beta parameter is the scale parameter of the distribution.
fn get_beta[P: Int](ez: Float64) -> Float64:
    """SIMD-optimized beta constant calculation using vectorized polynomial evaluation.
    """

    @parameter
    if 4 <= P <= 16:
        var zl = log2(ez + 1)
        var zl2 = zl * zl
        var zl3 = zl2 * zl
        var zl4 = zl3 * zl
        var zl5 = zl4 * zl
        var zl6 = zl5 * zl
        var zl7 = zl6 * zl

        var z = SIMD[DType.float64, 8](ez, zl, zl2, zl3, zl4, zl5, zl6, zl7)

        @parameter
        if P == 4:
            var c = SIMD[DType.float64, 8](
                -0.582581413904517,
                -1.935300357560050,
                11.079323758035073,
                -22.131357446444323,
                22.505391846630037,
                -12.000723834917984,
                3.220579408194167,
                -0.342225302271235,
            )
            return (c * z).reduce_add()
        elif P == 5:
            var c = SIMD[DType.float64, 8](
                -0.7518999460733967,
                -0.9590030077748760,
                5.5997371322141607,
                -8.2097636999765520,
                6.5091254894472037,
                -2.6830293734323729,
                0.5612891113138221,
                -0.0463331622196545,
            )
            return (c * z).reduce_add()
        elif P == 6:
            var c = SIMD[DType.float64, 8](
                29.8257900969619634,
                -31.3287083337725925,
                -10.5942523036582283,
                -11.5720125689099618,
                3.8188754373907492,
                -2.4160130328530811,
                0.4542208940970826,
                -0.0575155452020420,
            )
            return (c * z).reduce_add()
        elif P == 7:
            var c = SIMD[DType.float64, 8](
                2.8102921290820060,
                -3.9780498518175995,
                1.3162680041351582,
                -3.9252486335805901,
                2.0080835753946471,
                -0.7527151937556955,
                0.1265569894242751,
                -0.0109946438726240,
            )
            return (c * z).reduce_add()
        elif P == 8:
            var c = SIMD[DType.float64, 8](
                1.00633544887550519,
                -2.00580666405112407,
                1.64369749366514117,
                -2.70560809940566172,
                1.39209980244222598,
                -0.46470374272183190,
                0.07384282377269775,
                -0.00578554885254223,
            )
            return (c * z).reduce_add()
        elif P == 9:
            var c = SIMD[DType.float64, 8](
                -0.09415657458167959,
                -0.78130975924550528,
                1.71514946750712460,
                -1.73711250406516338,
                0.86441508489048924,
                -0.23819027465047218,
                0.03343448400269076,
                -0.00207858528178157,
            )
            return (c * z).reduce_add()
        elif P == 10:
            var c = SIMD[DType.float64, 8](
                -0.25935400670790054,
                -0.52598301999805808,
                1.48933034925876839,
                -1.29642714084993571,
                0.62284756217221615,
                -0.15672326770251041,
                0.02054415903878563,
                -0.00112488483925502,
            )
            return (c * z).reduce_add()
        elif P == 11:
            var c = SIMD[DType.float64, 8](
                -0.432325553856025,
                -0.108450736399632,
                0.609156550741120,
                -0.0165687801845180,
                -0.0795829341087617,
                0.0471830602102918,
                -0.00781372902346934,
                0.000584268708489995,
            )
            return (c * z).reduce_add()
        elif P == 12:
            var c = SIMD[DType.float64, 8](
                -0.384979202588598,
                0.183162233114364,
                0.130396688841854,
                0.0704838927629266,
                -0.0089589397146453,
                0.0113010036741605,
                -0.00194285569591290,
                0.000225435774024964,
            )
            return (c * z).reduce_add()
        elif P == 13:
            var c = SIMD[DType.float64, 8](
                -0.41655270946462997,
                -0.22146677040685156,
                0.38862131236999947,
                0.45340979746062371,
                -0.36264738324476375,
                0.12304650053558529,
                -0.01701540384555510,
                0.00102750367080838,
            )
            return (c * z).reduce_add()
        elif P == 14:
            var c = SIMD[DType.float64, 8](
                -0.371009760230692,
                0.00978811941207509,
                0.185796293324165,
                0.203015527328432,
                -0.116710521803686,
                0.0431106699492820,
                -0.00599583540511831,
                0.000449704299509437,
            )
            return (c * z).reduce_add()
        elif P == 15:
            var c = SIMD[DType.float64, 8](
                -0.38215145543875273,
                -0.89069400536090837,
                0.37602335774678869,
                0.99335977440682377,
                -0.65577441638318956,
                0.18332342129703610,
                -0.02241529633062872,
                0.00121399789330194,
            )
            return (c * z).reduce_add()
        elif P == 16:
            var c = SIMD[DType.float64, 8](
                -0.37331876643753059,
                -1.41704077448122989,
                0.40729184796612533,
                1.56152033906584164,
                -0.99242233534286128,
                0.26064681399483092,
                -0.03053811369682807,
                0.00155770210179105,
            )
            return (c * z).reduce_add()
        # Unreachable
        alias num_registers = 1 << P
        return 0.7213 / (1.0 + 1.079 / Float64(num_registers))
    else:
        # For larger register counts, use the standard beta correction
        alias num_registers = 1 << P
        return 0.7213 / (1.0 + 1.079 / Float64(num_registers))
