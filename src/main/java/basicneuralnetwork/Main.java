package basicneuralnetwork;

import basicneuralnetwork.utilities.FileReaderAndWriter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

public class Main {
    private static final int MAX_EPOCHS = 400;
    private static double percentForTesting = 0.3;

    public static void main(String[] args) {

        NeuralNetwork nn = new NeuralNetwork(4, 1,4,3);

        ArrayList<double[]> csvData = FileReaderAndWriter.readCSVFile("src/data/iris.data");

        ArrayList<double[]> train = new ArrayList<>();
        ArrayList<double[]> test = new ArrayList<>();

        splitData(csvData, train, test, percentForTesting);

        for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            batchTrain(train, nn);
        }

        testNN(test,nn);
    }

    private static void testNN(ArrayList<double[]> test, NeuralNetwork nn) {
        // TODO: loop over each element of the test data
        // TODO: extract the feature vector from the row
        // TODO: run output = nn.guess(...) to get the networks predictions
        // TODO: find the index of the largest value in output.  That's the guess!
        // TODO: check if that matches the correct label.  If so add to "correct" counter
        // TODO: display % correct
    }

    private static void batchTrain(ArrayList<double[]> train, NeuralNetwork nn) {
        // TODO: loop over each row in train
        // TODO: extract the feature vector you want (for iris, it's the first 4 elements in each row)
        // TODO: construct the correct output vector (for iris it's a length 3 double array with 1 marked in the index
        //       for the correct label
        // TODO: run nn.train(...)
    }

    private static void splitData(ArrayList<double[]> csvData, ArrayList<double[]> train, ArrayList<double[]> test, double percentForTesting){
        Collections.shuffle(csvData);

        // TODO: calculate # of items from csvData that should get added to test
        // TODO: add correct # of rows from csvData to test
        // TODO: add ther est to train
    }

}
