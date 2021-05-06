package rsoi.lab3.microservices.front.entity.user;

import rsoi.lab3.microservices.front.entity.BaseEntity;

import javax.validation.constraints.NotEmpty;

public class Role extends BaseEntity {

    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
