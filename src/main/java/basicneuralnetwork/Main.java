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
        // ** warning **  test does NOT just contain the feature vector.  It's the entire row from the file.
        // make a feature vector that's all the numbers except the last one.
        // get the correct label, which is the last number in the row

        // TODO: loop over each element of the test data
        // TODO: extract the feature vector from the row
        // TODO: run output = nn.guess(...) to get the networks predictions
        // TODO: find the index of the largest value in output.  That's the guess!
        // TODO: check if that matches the correct label.  If so add to "correct" counter
        // TODO: display % correct
    }

    private static void batchTrain(ArrayList<double[]> train, NeuralNetwork nn) {
        // ** warning **  test does NOT just contain the feature vector.  It's the entire row from the file.
        // make a feature vector that's all the numbers except the last one.
        // get the correct label, which is the last number in the row

        // TODO: loop over each row in train
        // TODO: extract the feature vector you want (for iris, it's the first 4 elements in each row)
        // TODO: construct the correct output vector (for iris it's a length 3 double array with 1 marked in the index
        //       for the correct label
        // TODO: run nn.train(...)
    }

    /***
     * This method is passed an EMPTY train and test list.  csvData represents all the data.
     * You will randomly divide the data up and add it to train and test.
     * Use percentForTesting to determine what percent of the overall data should get added to test.
     * @param csvData all the data
     * @param train empty list to be filled with training data
     * @param test empty list to be filled with test data
     * @param percentForTesting overall percent to be added to test list
     */
    private static void splitData(ArrayList<double[]> csvData, ArrayList<double[]> train, ArrayList<double[]> test, double percentForTesting){
        Collections.shuffle(csvData);
        // TODO: calculate # of items from csvData that should get added to test
        // TODO: add correct # of rows from csvData to test
        // TODO: add ther est to train
    }
}
