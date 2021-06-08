package ru.bmstu.adapt.domain;

import lombok.Builder;
import lombok.Data;
import org.bson.types.Binary;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.format.annotation.DateTimeFormat;

import javax.persistence.Id;
import javax.validation.constraints.NotNull;
import java.time.LocalDateTime;

@Document(collection = "detection_objects")
@Data
@Builder
public class DetectionObject {

    @Id
    private String id;

    @Indexed
    private String userId;

    @Indexed
    private String detectionProcessId;

    @Indexed
    private Integer numFrame;

    @NotNull
    private String clazz;

    private Integer[] box;

    private Double score;

    private Integer numObject;

    private String image;

    @NotNull
    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
    private LocalDateTime createdDate;

}