package es.us.indices;
import weka.core.*;

import java.util.ArrayList;
import java.util.List;

import static java.util.List.of;


/**
 * Created by Josem on 17/02/2017.
 */
public class KMeansIndices  {



    private static DenseInstance makeInstance(double[] values) {
        DenseInstance i = new DenseInstance(1.0, values);
        return i;
    }
    public static Indice calcularDunn(List<Cluster> clusters, DistanceFunction distanceFunction) {

        double dunn = 0.0;
        double max = 0;
        double min = 0;


        long startTime = System.currentTimeMillis();
        try {
            for (Cluster cluster : clusters) {
                for (Instance punto : cluster.getInstances()) {
                    for (Cluster cluster2 : clusters) {
                        if (!cluster.equals(cluster2)) {
                            for (Instance punto2 : cluster.getInstances()) {
                                if (!punto.equals(punto2)) {
                                    double dist = distanceFunction.distance(punto, punto2);
                                    if (min != 0) {
                                        if (dist < min) {
                                            min = dist;
                                        }
                                    } else {
                                        min = dist;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (Cluster cluster : clusters) {
                for (Instance punto : cluster.getInstances()) {
                    for (Instance punto2 : cluster.getInstances()) {
                        if (!punto.equals(punto2)) {
                            double dist = distanceFunction.distance(punto, punto2);
                            if (dist > max) {
                                max = dist;
                            }
                        }

                    }
                }
            }
            //System.out.println("MINIMO: " + min);
            // System.out.println("MAXIMO: " + max);

            dunn = min / max;
        } catch (Exception e) {
            e.printStackTrace();
        }
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(dunn, elapsedTime);

    }

    public static Indice calcularBDDunn(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double bdDunn = 0.0;
        double max = 0;
        double min = 0;

        long startTime = System.currentTimeMillis();
        try {

            for (Cluster cluster : clusters) {
                for (Cluster cluster2 : clusters) {
                    if (!cluster.equals(cluster2) && cluster.getCentroide() != null && cluster2.getCentroide() != null) {
                        double dist = distanceFunction.distance(cluster.getCentroide(), cluster2.getCentroide());
                        if (min != 0) {
                            if (dist < min) {
                                min = dist;
                            }
                        } else {
                            min = dist;
                        }
                    }
                }
            }


            //get the maximum distance of the points to the centroid of the cluster they belong to
            for (Cluster cluster : clusters) {
                if (cluster.getCentroide() != null) {
                    for (Instance punto : cluster.getInstances()) {
                        double dist = distanceFunction.distance(punto, cluster.getCentroide());
                        if (dist > max) {
                            max = dist;
                        }
                    }
                }
            }
            //System.out.println("MINIMO: " + min);
            //System.out.println("MAXIMO: " + max);

            bdDunn = min / max;
        } catch (Exception e) {
            e.printStackTrace();
        }

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(bdDunn, elapsedTime);

    }

    public static Indice calcularSilhouette(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double silhouette;
        double a;
        double distA = 0;
        double b;
        double distB = 0;
        double cont;

        long startTime = System.currentTimeMillis();

        for (Cluster cluster : clusters) {
            for (Instance punto : cluster.getInstances()) {
                for (Cluster cluster2 : clusters) {
                    if (!cluster.equals(cluster2)) {
                        for (Instance punto2 : cluster.getInstances()) {
                            if (!punto.equals(punto2)) {
                                distB += distanceFunction.distance(punto, punto2);

                            }
                        }
                    }
                }
            }
        }
        b = distB / clusters.size();

        cont = 0;
        for (Cluster cluster : clusters) {
            for (Instance punto : cluster.getInstances()) {
                for (Instance punto2 : cluster.getInstances()) {
                    if (!punto.equals(punto2)) {
                        distA += distanceFunction.distance(punto, punto2);
                        cont++;
                    }
                }
            }
        }
        a = distA / clusters.size();
        //System.out.println("A: " + a);
        //System.out.println("B: " + b);

        silhouette = b - a / Math.max(a, b);
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(silhouette, elapsedTime);
    }

    public static Indice calcularBDSilhouette(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double silhouette;
        double a;
        double distA = 0;
        double b;
        double distB = 0;
        double cont = 0;

        long startTime = System.currentTimeMillis();

        for (Cluster cluster : clusters) {
            if (cluster.getCentroide() != null) {
                for (Cluster cluster2 : clusters) {
                    if (cluster2.getCentroide() != null) {
                        if (!cluster.equals(cluster2)) {
                            distB += distanceFunction.distance(cluster.getCentroide(), cluster2.getCentroide());
                            cont++;
                        }
                    }
                }
            }
        }

        b = distB / cont;

        cont = 0;
        for (Cluster cluster : clusters) {
            if (cluster.getCentroide() != null) {
                for (Instance punto : cluster.getInstances()) {
                    distA += distanceFunction.distance(punto, cluster.getCentroide());
                    cont++;
                }
            }
        }
        a = distA / cont;
        //System.out.println("A: " + a);
        //System.out.println("B: " + b);

        silhouette = b - a / Math.max(a, b);
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(silhouette, elapsedTime);
    }

    private Indice calcularDavidBouldin(List<Cluster> clusters, DistanceFunction distanceFunction) {
        int numberOfClusters = clusters.size();
        double david = 0.0;

        long startTime = System.currentTimeMillis();

        if (numberOfClusters == 1) {
            throw new RuntimeException(
                    "Impossible to evaluate Davies-Bouldin index over a single cluster");
        } else {
            // counting distances within
            double[] withinClusterDistance = new double[numberOfClusters];

            int i = 0;
            for (Cluster cluster : clusters) {
                for (Instance punto : cluster.getInstances()) {
                    withinClusterDistance[i] += distanceFunction.distance(punto, cluster.getCentroide());
                }
                withinClusterDistance[i] /= cluster.getInstances().size();
                i++;
            }


            double result = 0.0;
            double max = Double.NEGATIVE_INFINITY;

            try {
                for (i = 0; i < numberOfClusters; i++) {
                    //if the cluster is null
                    if (clusters.get(i).getCentroide() != null) {

                        for (int j = 0; j < numberOfClusters; j++)
                            //if the cluster is null
                            if (i != j && clusters.get(j).getCentroide() != null) {
                                double val = (withinClusterDistance[i] + withinClusterDistance[j])
                                        / distanceFunction.distance(clusters.get(i).getCentroide(), clusters.get(j).getCentroide());
                                if (val > max)
                                    max = val;
                            }
                    }
                    result = result + max;
                }
            } catch (Exception e) {
                System.out.println("Excepcion al calcular DAVID BOULDIN");
                e.printStackTrace();
            }
            david = result / numberOfClusters;
        }

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(david, elapsedTime);
    }

    public static Indice calcularCalinskiHarabasz(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double calinski = 0.0;
        double squaredInterCluter = 0;
        double aux;
        double cont = 0;

        long startTime = System.currentTimeMillis();

        try {
            for (Cluster cluster : clusters) {
                if (cluster.getCentroide() != null) {
                    for (Cluster cluster2 : clusters) {
                        if (cluster2.getCentroide() != null) {
                            if (!cluster.equals(cluster2)) {
                                aux = distanceFunction.distance(cluster.getCentroide(), cluster2.getCentroide());
                                squaredInterCluter += aux * aux;
                                cont++;
                            }
                        }
                    }
                }
            }

            calinski = calcularSquaredDistance(clusters,distanceFunction).getResultado() / (squaredInterCluter / cont);
        } catch (Exception e) {
            System.out.println("Excepcion al calcular CALINSKI");
            e.printStackTrace();
        }

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(calinski, elapsedTime);
    }

    //Diámetro máximo entre dos puntos que pertenecen al mismo cluster.
    public static Indice calcularMaximumDiameter(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double maximumDiameter = 0;
        double aux;

        long startTime = System.currentTimeMillis();

        for (Cluster cluster : clusters) {
            for (Instance punto : cluster.getInstances()) {
                for (Instance punto2 : cluster.getInstances()) {
                    if (!punto.equals(punto2)) {
                        aux = distanceFunction.distance(punto, punto2);
                        if (aux > maximumDiameter) {
                            maximumDiameter = aux;
                        }
                    }
                }
            }
        }

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(maximumDiameter, elapsedTime);
    }

    //Media de distancia cuadrática entre los puntos del mismo cluster.
    public static Indice calcularSquaredDistance(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double squaredDistance = 0;
        double aux;
        double cont = 0;

        long startTime = System.currentTimeMillis();

        for (Cluster cluster : clusters) {
            for (Instance punto : cluster.getInstances()) {
                for (Instance punto2 : cluster.getInstances()) {
                    if (!punto.equals(punto2)) {
                        aux = distanceFunction.distance(punto, punto2);
                        squaredDistance += aux * aux;
                        cont++;
                    }
                }
            }
        }

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(squaredDistance / cont, elapsedTime);
    }

    public static Indice calcularAverageDistance(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double averageDistance;
        double distA = 0;
        double cont = 0;

        long startTime = System.currentTimeMillis();

        for (Cluster cluster : clusters) {
            for (Instance punto : cluster.getInstances()) {
                for (Instance punto2 : cluster.getInstances()) {
                    if (!punto.equals(punto2)) {
                        distA += distanceFunction.distance(punto, punto2);
                        cont++;
                    }
                }
            }
        }
        averageDistance = distA / cont;


        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(averageDistance, elapsedTime);
    }

     public static Indice calcularAverageBetweenClusterDistance(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double averageDistanceBetween;
        double distA = 0;
        double cont = 0;

        long startTime = System.currentTimeMillis();

        for (Cluster cluster : clusters) {
            for (Instance punto : cluster.getInstances()) {
                for (Cluster cluster2 : clusters) {
                    if (!cluster.equals(cluster2)) {
                        for (Instance punto2 : cluster.getInstances()) {
                            if (!punto.equals(punto2)) {
                                distA += distanceFunction.distance(punto, punto2);
                                cont++;
                            }
                        }
                    }
                }
            }
        }
        averageDistanceBetween = distA / cont;

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(averageDistanceBetween, elapsedTime);

    }

    //Distancia minima entre puntos de diferentes clusters
    public static Indice calcularMinimumDistance(List<Cluster> clusters, DistanceFunction distanceFunction) {
        double minimumDistance = -1;
        double aux;

        long startTime = System.currentTimeMillis();

        for (Cluster cluster : clusters) {
            for (Instance punto : cluster.getInstances()) {
                for (Cluster cluster2 : clusters) {
                    if (!cluster.equals(cluster2)) {
                        for (Instance punto2 : cluster.getInstances()) {
                            if (!punto.equals(punto2)) {
                                if (minimumDistance == -1) {
                                    minimumDistance = distanceFunction.distance(punto, punto2);
                                } else {
                                    aux = distanceFunction.distance(punto, punto2);
                                    if (aux < minimumDistance)
                                        minimumDistance = aux;
                                }
                            }
                        }
                    }
                }
            }
        }
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        return new Indice(minimumDistance, elapsedTime);
    }



    public static void main(String[] args) {
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        attributes.add(new Attribute("a1"));
        attributes.add(new Attribute("a2"));
        attributes.add(new Attribute("a3"));

        Instances instances = new Instances("test", attributes, 3);
        instances.add(makeInstance(new double[]{-1,2,3}));
        instances.add(makeInstance(new double[]{4,0,-3}));
        instances.add(makeInstance(new double[]{6,2,3}));
        instances.add(makeInstance(new double[]{-10,20,30}));
        instances.add(makeInstance(new double[]{40,0,-30}));
        instances.add(makeInstance(new double[]{60,20,30}));

        Cluster cla = new Cluster();
        cla.getInstances().addAll(instances.subList(0,2));
        cla.setCentroide(instances.get(1));

        Cluster clb = new Cluster();
        clb.getInstances().addAll(instances.subList(3,5));
        clb.setCentroide(instances.get(4));


        DistanceFunction distanceFunction = new EuclideanDistance(instances);


        List<Cluster> clusters= new ArrayList<Cluster>();

        clusters.add(cla);
        clusters.add(clb);

        System.out.println(calcularDunn(clusters,distanceFunction));
    }
}



