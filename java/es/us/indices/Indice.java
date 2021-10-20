package es.us.indices;

/**
 * Created by Josem on 21/04/2017.
 */
public class Indice {

    private final double resultado;
    private final long time;

    public Indice(double resultado, long time) {
        this.resultado = resultado;
        this.time = time;
    }

    public double getResultado() {
        return resultado;
    }


}
