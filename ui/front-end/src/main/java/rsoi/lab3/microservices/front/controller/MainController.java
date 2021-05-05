package rsoi.lab3.microservices.front.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;
import rsoi.lab3.microservices.front.entity.user.User;
import rsoi.lab3.microservices.front.model.TaskPage;
import rsoi.lab3.microservices.front.service.TaskService;
import rsoi.lab3.microservices.front.service.jwt.JwtTokenProvider;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@Controller
public class MainController {

    @Autowired
    private TaskService taskService;
    @Autowired
    private JwtTokenProvider jwtTokenProvider;

    @GetMapping(value = "/")
    public ModelAndView index(@RequestParam(value = "page") Optional<Integer> page,
                              @RequestParam(value = "size") Optional<Integer> size,
                              @CookieValue(value = "ut", required = false) Cookie cookie,
                              HttpServletRequest request,
                              HttpServletResponse response,
                              Model model) {
        int currentPage = page.orElse(0);
        int pageSize = size.orElse(9);
        request.getSession().setAttribute("resultTest", null);
        User user;
        if (cookie == null || cookie.getValue().equals(""))
            user = new User();
        else if (jwtTokenProvider.validateAccessToken(cookie.getValue()))
            user = jwtTokenProvider.getUserByToken(cookie.getValue());
        else {
            request.getSession().setAttribute("resultTest", null);
            model.addAttribute("user", new User());
            request.getSession().invalidate();
            cookie.setValue(null);
            cookie.setPath("/");
            response.addCookie(cookie);
            user = new User();
        }
        model.addAttribute("user", user);
        TaskPage tasksPage = taskService.findAll(currentPage, pageSize);
        model.addAttribute("tasksPage", tasksPage);
        return new ModelAndView("index");
    }

    @GetMapping(value = "/image/{idImage}", produces = MediaType.APPLICATION_OCTET_STREAM_VALUE)
    @ResponseBody
    public byte[] getImage(@PathVariable Long idImage, HttpServletRequest request) throws IOException {
        request.getSession().setAttribute("resultTest", null);
        Path path = Paths.get(System.getProperty("user.dir") + File.separator +
                "files" + File.separator + "image" + File.separator + idImage + ".png");
        return Files.readAllBytes(path);
    }
}
