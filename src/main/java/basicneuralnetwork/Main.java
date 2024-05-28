package basicneuralnetwork;

import basicneuralnetwork.activationfunctions.ActivationFunction;
import basicneuralnetwork.utilities.FileReaderAndWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class Main {
    private static final int MAX_EPOCHS = 4000;
    private static double percentForTesting = 0.4;

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(4, 3,20,3);
        nn.setLearningRate(0.01);
        //nn.setActivationFunction(ActivationFunction.RELU);

        ArrayList<double[]> csvData = FileReaderAndWriter.readCSVFile("src/data/iris.data");

        ArrayList<double[]> train = new ArrayList<>();
        ArrayList<double[]> test = new ArrayList<>();

        splitData(csvData, train, test, percentForTesting);

        System.out.println("Train error \t\t test error");
        for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            batchTrain(train, nn);
            int[] results = {0, 0};
            double testError = testNN(test,nn, results);
            double trainError = testNN(train, nn);

            if (epoch % 10 == 0) {
                System.out.println(trainError + "\t\t" + testError + "\t\t" + results[0] + "/" + results[1]);
            }
        }
    }

    private static double testNN(ArrayList<double[]> test, NeuralNetwork nn) {
        return testNN(test, nn, null);
    }

    private static double testNN(ArrayList<double[]> test, NeuralNetwork nn, int[] results) {
        double errorSum = 0;

        if (results != null) {
            results[0] = 0;
            results[1] = 0;
        }

        for (double[] row : test) {
            double[] input = Arrays.copyOfRange(row, 0, row.length-1);
            int label = (int)(row[row.length-1]);
            double[] correctOutput = getOutputVectorFor(label);
            double[] output = nn.guess(input);
            output = softMax(output);
            errorSum += logLossError(output, correctOutput);

            int guess = getLargestIndex(output);
            if (results != null) {
                if (guess == label) {
                    results[0]++;
                }
                results[1]++;
            }
        }

        return errorSum/test.size();
    }

    public static double[] softMax(double[] input) {
        double[] output = new double[input.length];
        double max = input[0];

        for (int i = 1; i < input.length; i++) {
            if (input[i] > max) {
                max = input[i];
            }
        }

        double sum = 0.0;

        // Compute the exponentials and their sum
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i] - max);
            sum += output[i];
        }

        // Normalize the exponentials to get the probabilities
        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    private static double logLossError(double[] guess, double[] actual) {

        for (int i = 0; i < actual.length; i++) {
            if (Math.abs(actual[i]-1.0) < 0.0001) {
                return -Math.log(guess[i]);
            }
        }

        System.err.println("Error: no output class close to 1.0 in actual array in logLosError");
        return Double.MAX_VALUE;
    }

    private static int getLargestIndex(double[] output) {
        int index = 0;

        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[index]) {
                index = i;
            }
        }

        return index;
    }

    private static void batchTrain(ArrayList<double[]> train, NeuralNetwork nn) {
        for (double[] row : train) {
            double[] input = Arrays.copyOfRange(row, 0, row.length-1);
            int label = (int)(row[row.length-1]);
            double[] correctOutput = getOutputVectorFor(label);

            nn.train(input, correctOutput);
        }
    }

    private static double[] getOutputVectorFor(int label) {
        double[] correctOutput = new double[] {0,0,0};
        correctOutput[label]++;
        return correctOutput;
    }

    private static void splitData(ArrayList<double[]> csvData, ArrayList<double[]> train, ArrayList<double[]> test, double percentForTesting){
        Collections.shuffle(csvData);
        int numTest = (int)(percentForTesting*csvData.size());

        for (int i = 0; i < numTest; i++) {
            test.add(csvData.remove(0));
        }

        while (csvData.size() > 0) {
            train.add(csvData.remove(0));
        }
    }

}
