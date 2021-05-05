package rsoi.lab3.microservices.front.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;
import rsoi.lab3.microservices.front.entity.task.Task;
import rsoi.lab3.microservices.front.entity.test.Test;
import rsoi.lab3.microservices.front.entity.user.User;
import rsoi.lab3.microservices.front.exception.feign.ClientAuthenticationExceptionWrapper;
import rsoi.lab3.microservices.front.model.TaskPage;
import rsoi.lab3.microservices.front.service.TaskService;
import rsoi.lab3.microservices.front.service.jwt.JwtTokenProvider;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.UUID;

@Controller
public class TaskChangeController {

    @Autowired
    private TaskService taskService;
    @Autowired
    private JwtTokenProvider jwtTokenProvider;
    private static final Logger log = LoggerFactory.getLogger(TaskChangeController.class);

    @GetMapping(value = "/auth/task")
    public String tasksByUser(Model model,
                              @RequestParam(value = "page") Optional<Integer> page,
                              @RequestParam(value = "size") Optional<Integer> size, HttpServletRequest request,
                              @CookieValue("ut") Cookie cookie, HttpServletResponse response) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";

        int currentPage = page.orElse(0);
        int pageSize = size.orElse(9);
        request.getSession().setAttribute("resultTest", null);
        User u = jwtTokenProvider.getUserByToken(cookie.getValue());
        request.getSession().setAttribute("task", null);
        if (u == null) {
            u = new User();
        } else {
            try {
                TaskPage tasksPage = taskService.findByUserId(u.getId(), currentPage, pageSize, jwtTokenProvider.resolveToken(cookie));
                model.addAttribute("tasksPage", tasksPage);
            } catch (ClientAuthenticationExceptionWrapper exc) {
                request.getSession().setAttribute("resultTest", null);
                model.addAttribute("user", new User());
                request.getSession().invalidate();
                cookie.setValue(null);
                cookie.setPath("/");
                response.addCookie(cookie);
                return "redirect:/";
            }
        }
        model.addAttribute("uuid", new UUID(0, 0));
        model.addAttribute("user", u);
        return "tasks_user";
    }

    @GetMapping(value = "/auth/task/{idTask}")
    public String taskByUserIdAndTaskId(Model model, HttpServletRequest request, @CookieValue("ut") Cookie cookie, @PathVariable UUID idTask) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";

        request.getSession().setAttribute("resultTest", null);
        User u = jwtTokenProvider.getUserByToken(cookie.getValue());
        Task t = null;
        model.addAttribute("task", t);
        model.addAttribute("zeroUUID", new UUID(0,0));
        if (u == null) {
            u = new User();
        } else {
            try {
                t = taskService.findByUserIdAndTaskId(idTask, cookie.getValue());
                if (t != null)
                    request.getSession().setAttribute("image", t.getImage());
            } catch (Exception exc) {
                log.error(exc.getMessage());
            }
        }
        if (t == null) {
            t = new Task();
            t.setTest(new Test());
            t.setWithoutTest((byte) 0);
        }
        model.addAttribute("task", t);
        model.addAttribute("user", u);
        return "task_change";
    }

    @PostMapping(value = "/auth/task/{idTask}/upload")
    public String upload(@PathVariable UUID idTask, @RequestBody MultipartFile file,
                         Model model, HttpServletRequest request, @CookieValue(value = "ut", required = false) Cookie cookie) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";

        User u = jwtTokenProvider.getUserByToken(cookie.getValue());
        if (u != null) {
            if (!file.isEmpty()) {
                String directory = System.getProperty("user.dir") + File.separator + "files" + File.separator + "image" + File.separator;
                String name = directory + idTask + ".png";
                try {
                    Files.deleteIfExists(Paths.get(name));
                    Files.createDirectories(Paths.get(directory));
                    Files.copy(file.getInputStream(), Paths.get(name));
                    request.getSession().setAttribute("image", idTask + "");
                } catch (Exception e) {
                    return String.format("redirect:/auth/task/%s", idTask);
                }
            }
            return String.format("redirect:/auth/task/%s", idTask);
        }
        return "redirect:/";
    }

    @PostMapping(value = "/auth/task/create")
    public String createTask(@Valid @RequestBody Task task,
                             Model model, HttpServletRequest request, @CookieValue(value = "ut", required = false) Cookie cookie) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";

        User u = jwtTokenProvider.getUserByToken(cookie.getValue());
        if (u != null) {
            String image = (String) request.getSession().getAttribute("image");
            task.setIdUser(u.getId());
            if (image != null)
                task.setImage(image);
            task.setId(null);
            task = taskService.create(task, cookie.getValue());
            if (task != null)
                return String.format("redirect:/auth/task/%s", task.getId());
            return "redirect:/auth/task";
        }
        return "redirect:/";
    }

    @PostMapping(value = "/auth/task/{idTask}/update")
    public String updateTask(@PathVariable UUID idTask,
                             @Valid @RequestBody Task task, Model model, HttpServletRequest request,
                             @CookieValue(value = "ut", required = false) Cookie cookie) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";

        User u = jwtTokenProvider.getUserByToken(cookie.getValue());
        if (u != null) {
            String image = (String) request.getSession().getAttribute("image");
            if (image != null)
                task.setImage(image);
            taskService.update(idTask, task, cookie.getValue());
            return "redirect:/auth/task";
        }
        return "redirect:/";
    }

    @PostMapping(value = "/auth/task/{idTask}/delete")
    public String deleteTask(@PathVariable UUID idTask,
                             Model model, HttpServletRequest request, @CookieValue(value = "ut", required = false) Cookie cookie) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";

        User u = jwtTokenProvider.getUserByToken(cookie.getValue());
        if (u != null) {
            taskService.delete(idTask, cookie.getValue());
            return "redirect:/auth/task";
        }
        return "redirect:/";
    }

}
