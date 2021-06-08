package ru.bmstu.adapt.model;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class DetectionObject {

    private String id;
    private String userId;
    private String detectionProcessId;
    private Integer numFrame;
    private String clazz;
    private Integer[] box;
    private Double score;
    private Integer numObject;
    private String image;
    private LocalDateTime createdDate;

}