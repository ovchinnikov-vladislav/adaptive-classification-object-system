package bmstu.dynamic.simulator.dto;

import lombok.Data;

@Data
public class DetectionObjectRequest {

    private String clazz;
    private Integer[] box;
    private Double score;
    private Integer numObject;
    private Integer iteration;
    private String image;

}
