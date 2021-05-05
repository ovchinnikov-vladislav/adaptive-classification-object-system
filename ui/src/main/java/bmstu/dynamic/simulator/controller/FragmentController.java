package bmstu.dynamic.simulator.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/fragment")
@RequiredArgsConstructor
public class FragmentController {

    @GetMapping("/sidebar")
    public String sidebar() {
        return "fragments/sidebar";
    }

    @GetMapping("/logout")
    public String logout() {
        return "fragments/logout";
    }

    @GetMapping("/footer")
    public String footer() {
        return "fragments/footer";
    }

    @GetMapping("/topbar")
    public String topbar() {
        return "fragments/topbar";
    }

}
