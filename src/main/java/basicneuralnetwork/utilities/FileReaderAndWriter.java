package basicneuralnetwork.utilities;

import basicneuralnetwork.NeuralNetwork;
import basicneuralnetwork.activationfunctions.ActivationFunction;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;
import org.ejml.data.Matrix;
import org.ejml.simple.SimpleOperations;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by KimFeichtinger on 26.04.18.
 */
public class FileReaderAndWriter {

    public static ArrayList<double[]> readCSVFile(String filePath) {
        ArrayList<double[]> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");

                double[] row = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    if (!values[i].trim().equals("")) {
                        row[i] = Double.parseDouble(values[i]);
                    }
                }
                data.add(row);

            }
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Check if your csv file contains values that aren't doubles.");
        }

        return data;
    }

    public static void writeToFile(NeuralNetwork nn, String fileName) {
        String name = fileName;

        if (fileName == null) {
            name = "nn_data";
        }

        try {
            FileWriter file = new FileWriter(name + ".json");
            Gson gson = getGsonBuilder().create();
            String nnData = gson.toJson(nn);

            file.write(nnData);
            file.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static NeuralNetwork readFromFile(String fileName) {
        NeuralNetwork nn = null;
        String name = fileName;

        if (fileName == null) {
            name = "nn_data.json";
        }

        try {
            Gson gson = getGsonBuilder().create();
            JsonReader jsonReader = new JsonReader(new FileReader(name));
            nn = gson.fromJson(jsonReader, NeuralNetwork.class);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return nn;
    }

    // Get a GsonBuilder-object with all the needed TypeAdapters added
    private static GsonBuilder getGsonBuilder() {
        GsonBuilder gsonBuilder = new GsonBuilder();

        gsonBuilder.registerTypeAdapter(ActivationFunction.class, new InterfaceAdapter<ActivationFunction>());
        gsonBuilder.registerTypeAdapter(Matrix.class, new InterfaceAdapter<Matrix>());
        gsonBuilder.registerTypeAdapter(SimpleOperations.class, new InterfaceAdapter<SimpleOperations>());
        gsonBuilder.setPrettyPrinting();

        return gsonBuilder;
    }

}
