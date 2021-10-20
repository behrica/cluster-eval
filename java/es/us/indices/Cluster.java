package es.us.indices;

import weka.core.Instance;

import java.util.ArrayList;
import java.util.List;

public class Cluster {
    private List<Instance> puntos = new ArrayList<Instance>();
    private Instance centroide;

    public Instance getCentroide() {
        return centroide;
    }

    public void setCentroide(Instance centroide) {
        this.centroide = centroide;
    }

    public List<Instance> getInstances() {
        return puntos;
    }


}
