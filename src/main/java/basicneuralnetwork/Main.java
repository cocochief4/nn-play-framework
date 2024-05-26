package basicneuralnetwork;

import basicneuralnetwork.utilities.FileReaderAndWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class Main {
    private static final int MAX_EPOCHS = 400;
    private static double percentForTesting = 0.4;

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(4, 1,4,3);

        ArrayList<double[]> csvData = FileReaderAndWriter.readCSVFile("src/data/iris.data");

        ArrayList<double[]> train = new ArrayList<>();
        ArrayList<double[]> test = new ArrayList<>();

        splitData(csvData, train, test, percentForTesting);

        for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            batchTrain(train, nn);
            testNN(test,nn);
        }
    }

    private static void testNN(ArrayList<double[]> test, NeuralNetwork nn) {
        int correct = 0;
        int total = 0;

        for (double[] row : test) {
            double[] input = Arrays.copyOfRange(row, 0, row.length-1);
            int label = (int)(row[row.length-1]);

            double[] output = nn.guess(input);
            int guess = getLargestIndex(output);
            if (guess == label) {
                correct++;
            }
            total++;

            System.out.println("correct: " + correct + "/" + total);
        }
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
            double[] correctOutput = new double[] {0,0,0};
            correctOutput[label]++;

            nn.train(input, correctOutput);
        }
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
