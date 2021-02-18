import matplotlib.pyplot as plt
import numpy as np
import math
    
# INPUT:
# inference_results_confidence : a vector of inference confidences
# expected_results : a vector of corisponding ground truth values
# amount_of_bins : the amount of bins for the diagram (default=10)
# custom : True, to use a custom reliability diagram (default=false)

def reliability_diagram(inference_results_confidence, expected_results, amount_of_bins = 10, custom = False):

    if (len(inference_results_confidence)!=len(expected_results)):
        print("Input size mismatch: " )
        print("len(inference_results_confidence) = " + str(len(inference_results_confidence)))
        print("len(expected_results) = " + str(len(expected_results)))
        return None
    
    # Init lists
    bins = [0] * amount_of_bins             # count true score results per bin
    binsTotal = [0] * amount_of_bins        # count all score results per bin
    sumScorePerBin = [0] * amount_of_bins   # sum all score results per bin
    labels = [""] * amount_of_bins          # define labels for bin seperation
    emptyBins = [0] * amount_of_bins        # mark empty bins
    MCEtemp = [0] * amount_of_bins          # temp variable, used in calculating MCE

    step = 1 / amount_of_bins

    # Populate bins
    for i , value in enumerate(inference_results_confidence):
        if ((value > 1) or (value < 0)):
            print("Wrong confidence value " + str(value) + ",in pos " + str(i))
            return None
        binsTotal[math.ceil(value / step) - 1] += 1 
        sumScorePerBin[math.ceil(value / step) - 1] += value 
        if (expected_results[i] == 1):
            bins[math.ceil(value / step) - 1] += 1  
    
    # Check for empty bins
    for idx, value in enumerate(binsTotal):
        if (value == 0):
            binsTotal[idx] = 1
            emptyBins[idx] = 1

    # Calculate bar-bins
    x = np.arange(0 ,amount_of_bins ,1)
    res = [(i)/ (j) for i, j in zip(bins, binsTotal)]
    tempAvgScore = [(i)/ (j) for i, j in zip(sumScorePerBin, binsTotal)]
    AverageScore = np.asarray(tempAvgScore)   # S_m in paper
    Accuracy = np.asarray(res)                  # A_m in paper
    xAxis = (10 * (1 + x)) / amount_of_bins

    # Draw bins
    if (not custom):
        plt.figure(1)
        plt.bar(10 * AverageScore, Accuracy, color='green', width=0.5)
        plt.plot([0,10], [0,1], 'r--')
        plt.title("Average Calibration Error")
        for ind, a in enumerate(AverageScore):
            labels[ind] = math.floor(a*100)/100

        plt.xticks(10 * AverageScore, labels)
    else:
        plt.figure(2)
        plt.bar(xAxis, Accuracy, color='green', width=0.5)
        plt.bar(xAxis, emptyBins, color='lightgray', width=0.5)
        plt.plot([xAxis[0],10], [0.1,1], 'r--')
        plt.xlabel("Confidence bins")
        plt.ylabel("Accuracy")
        plt.title("Custom Calibration Error")
        for ind, a in enumerate(xAxis):
            if (ind == 0) :
                labels[ind] = str(0.0) + " -\n" + str(math.floor(100*a)/1000)
            else:
                labels[ind] = str(math.floor(100*(xAxis[ind-1]))/1000) + " -\n" + str(math.floor(100*a)/1000)

        plt.xticks(xAxis, labels)
    
    M = amount_of_bins - sum(emptyBins)
    ACE = 0
    for ind, value in enumerate(AverageScore):
        ACE +=(abs(AverageScore[ind] - Accuracy[ind])/M)
        MCEtemp[ind] = (abs(AverageScore[ind] - Accuracy[ind]))
    MCE = max(MCEtemp)

    print("ACE: " + str(round(ACE*100,2)) + "%")
    print("MCE: " + str(round(MCE*100,2)) + "%")
    plt.show()


if __name__ == "__main__":

    # perfect linear example
    inference_results_confidence = [0.92, 0.91, 0.93, 0.91, 0.92, 0.901, 0.92, 0.98, 0.92, 0.95,
                    0.86, 0.84, 0.82, 0.81, 0.875, 0.83, 0.809, 0.83, 0.88, 0.82,
                    0.721, 0.77, 0.706, 0.716, 0.73, 0.79, 0.72, 0.703, 0.71, 0.8,
                    0.61, 0.62, 0.63, 0.64, 0.655, 0.63, 0.67, 0.68, 0.69, 0.66,
                    0.503, 0.54, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51,
                    0.462, 0.402, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41,
                    0.357, 0.336, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31,
                    0.205, 0.254, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21,
                    0.193, 0.142, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11,
                    0.043, 0.021, 0.012, 0.02, 0.04, 0.05, 0.04, 0.03, 0.02, 0.01]
    expected_results = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    reliability_diagram(inference_results_confidence, expected_results)