package rsoi.lab3.microservices.front.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import rsoi.lab3.microservices.front.entity.task.Task;

import java.util.ArrayList;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class TaskPage extends PageImpl<Task> {

    @JsonCreator(mode = JsonCreator.Mode.PROPERTIES)
    public TaskPage(@JsonProperty("content") List<Task> content,
                    @JsonProperty("number") int number,
                    @JsonProperty("size") int size,
                    @JsonProperty("totalElements") Long totalElements,
                    @JsonProperty("pageable") JsonNode pageable,
                    @JsonProperty("last") boolean last,
                    @JsonProperty("totalPages") int totalPages,
                    @JsonProperty("sort") JsonNode sort,
                    @JsonProperty("first") boolean first,
                    @JsonProperty("numberOfElements") int numberOfElements) {

        super(content, PageRequest.of(number, size), totalElements);
    }

    public TaskPage(List<Task> content, Pageable pageable, long total) {
        super(content, pageable, total);
    }

    public TaskPage(List<Task> content) {
        super(content);
    }

    public TaskPage() {
        super(new ArrayList<>());
    }
}
