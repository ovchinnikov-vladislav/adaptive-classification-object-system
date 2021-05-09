package bmstu.dynamic.simulator.dto;

import lombok.Data;

@Data
public class ClassificationDataRequest {

    private String clazz;
    private Integer[] box;
    private Double score;
    private Integer num;

}
