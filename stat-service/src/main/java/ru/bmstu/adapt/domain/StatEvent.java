package ru.bmstu.adapt.domain;

import lombok.Data;

import javax.validation.constraints.NotNull;
import java.util.HashMap;
import java.util.Map;

@Data
public class StatEvent {

    @NotNull
    private StatEventType type;

    @NotNull
    private Map<String, ?> attributes = new HashMap<>();

}
