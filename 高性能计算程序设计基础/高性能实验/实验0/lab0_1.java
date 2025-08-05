import java.util.Random;

class Matrix {
    private int rows;
    private int cols;
    private float[][] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols];
    }

    public void generateRandomValues(Random rand) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rand.nextFloat();
            }
        }
    }

    public void multiply(Matrix B, Matrix C) {
        if (this.cols != B.rows) {
            throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");
        }

        for (int p = 0; p < C.cols; p++) {
            for (int l = 0; l < this.cols; l++) {
                for (int i = 0; i < C.rows; i++) {
                    C.data[i][p] += this.data[i][l] * B.data[l][p];
                }
            }
        }
    }

    public void print(int limitRows, int limitCols) {
        int displayRows = Math.min(rows, limitRows);
        int displayCols = Math.min(cols, limitCols);

        for (int i = 0; i < displayRows; i++) {
            for (int j = 0; j < displayCols; j++) {
                System.out.printf("%f ", data[i][j]);
            }
            if (cols > displayCols) {
                System.out.print("...");
            }
            System.out.println();
        }
        if (rows > displayRows) {
            System.out.println("...");
        }
    }
}

public class lab0_1 {
    private static final int M = 1200;
    private static final int N = 1000;
    private static final int K = 800;

    public static void main(String[] args) {
        Matrix A = new Matrix(M, N);
        Matrix B = new Matrix(N, K);
        Matrix C = new Matrix(M, K);

        Random rand = new Random();

        A.generateRandomValues(rand);
        B.generateRandomValues(rand);

        System.out.println("Calculating running time...");
        long startTime = System.currentTimeMillis();

        A.multiply(B, C);

        long endTime = System.currentTimeMillis();
        double timeTaken = (endTime - startTime) / 1000.0;

        System.out.println("\nMatrix A (first 2*2 elements):");
        A.print(2, 2);

        System.out.println("\nMatrix B (first 2*2 elements):");
        B.print(2, 2);

        System.out.println("\nMatrix C (Result of A*B, first 2*2 elements):");
        C.print(2, 2);

        System.out.println("\nRun time: " + timeTaken + " seconds.\n");
    }
}
