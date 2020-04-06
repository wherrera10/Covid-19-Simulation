# Simplistic Julia extrapolation model to fit world data totals

using LsqFit

const K = 7_800_000_000  # approximate world population
const n0 = 27  # starting at day 0 with 27 Chinese cases

""" The model for logistic regression with a given r0 """
@. model(t, r0) = (n0 * exp(r0 * t)) / (( 1 + n0 * (exp(r0 * t) - 1) / K))

# Source: https://ourworldindata.org/coronavirus#all-charts-preview
# Daily world totals, including China and all countries
ydata = [
27, 27, 27, 44, 44, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60,
61, 61, 66, 83, 219, 239, 392, 534, 631, 897, 1350, 2023,
2820, 4587, 6067, 7823, 9826, 11946, 14554, 17372, 20615,
24522, 28273, 31491, 34933, 37552, 40540, 43105, 45177,
60328, 64543, 67103, 69265, 71332, 73327, 75191, 75723,
76719, 77804, 78812, 79339, 80132, 80995, 82101, 83365,
85203, 87024, 89068, 90664, 93077, 95316, 98172, 102133,
105824, 109695, 114232, 118610, 125497, 133852, 143227,
151367, 167418, 180096, 194836, 213150, 242364, 271106,
305117, 338133, 377918, 416845, 468049, 527767, 591704,
656866, 715353, 777796, 851308, 928436, 1000249, 1082054,
1174652,
]
# The initial data is for 97 days starting with day 0 = 31 December 2019
tdata = collect(LinRange(0.0, 96, 97))

# starting approximation for r0 of 1/2
r0param = [0.5]

fit = curve_fit(model, tdata, ydata, r0param)

# Our answer for r0 given the world data and simplistic model
R0 = fit.param
println("The R0 for the world data is: ", R0)
println("The confidence interval at 5% significance is: ",
    confidence_interval(fit, 0.05))

# output --> The R0 for the world data is: [0.11230217572265622]

