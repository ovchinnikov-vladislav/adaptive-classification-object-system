package rsoi.lab3.microservices.front.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;
import rsoi.lab3.microservices.front.entity.result.Result;
import rsoi.lab3.microservices.front.entity.task.Task;
import rsoi.lab3.microservices.front.entity.user.User;
import rsoi.lab3.microservices.front.model.ExecuteTaskRequest;
import rsoi.lab3.microservices.front.model.ResultTest;
import rsoi.lab3.microservices.front.service.TaskExecutorService;
import rsoi.lab3.microservices.front.service.TaskService;
import rsoi.lab3.microservices.front.service.jwt.JwtTokenProvider;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;
import java.util.UUID;

@Controller
public class TaskExecuteController {

    @Autowired
    private TaskExecutorService executorService;
    @Autowired
    private TaskService taskService;
    @Autowired
    private JwtTokenProvider jwtTokenProvider;

    @GetMapping(value = "/auth/execute_task/{id}")
    public String task(@PathVariable UUID id, Model model, HttpServletRequest request,
                       @CookieValue("ut") Cookie cookie, HttpServletResponse response) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";
        User u = jwtTokenProvider.getUserByToken(cookie.getValue());

        ResultTest resultTest = (ResultTest) request.getSession().getAttribute("resultTest");
        Result result = (Result) request.getSession().getAttribute("result");
        Task t = new Task();
        if (u == null) {
            u = new User();
        } else {
            t = taskService.findById(id);
        }
        model.addAttribute("user", u);
        model.addAttribute("task", t);
        model.addAttribute("resultTest", resultTest);
        model.addAttribute("result", result);
        request.getSession().setAttribute("resultTest", null);
        request.getSession().setAttribute("result", null);
        if (t == null)
            return "404";
        return "task";
    }

    @PostMapping(value = "/auth/execute_task/{id}")
    public String sendTaskOnExecute(@Valid @ModelAttribute Task task, @PathVariable UUID id,
                                    Model model, HttpServletRequest request,
                                    @CookieValue("ut") Cookie cookie, HttpServletResponse response) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";
        User u = jwtTokenProvider.getUserByToken(cookie.getValue());
        Task t = (Task) request.getSession().getAttribute("task");
        if (u != null && t != null) {
            ExecuteTaskRequest executeTaskRequest = new ExecuteTaskRequest();
            executeTaskRequest.setIdTask(t.getId());
            executeTaskRequest.setIdUser(t.getIdUser());
            executeTaskRequest.setSourceTask(task.getTemplateCode());
        } else {
            u = new User();
            t = new Task();
        }
        return "redirect:/auth/execute_task/" + task.getId();
    }

    @PostMapping(value = "/auth/execute_task/execute")
    public String executeTask(@Valid @RequestBody ExecuteTaskRequest executeTaskRequest, @CookieValue("ut") Cookie cookie, HttpServletRequest request) {
        if (cookie == null)
            return "redirect:/oauth/login";
        if (cookie.getValue().equals(""))
            return "redirect:/oauth/login";
        if (!jwtTokenProvider.validateAccessToken(cookie.getValue()))
            return "redirect:/oauth/login";

        ResultTest resultTest = executorService.execute(executeTaskRequest, cookie.getValue());
        request.getSession().setAttribute("resultTest", resultTest);
        Result result = executorService.findResultByUserIdAndTaskId(executeTaskRequest.getIdUser(), executeTaskRequest.getIdTask(), cookie.getValue());
        request.getSession().setAttribute("result", result);
        return "redirect:/auth/execute_task/" + executeTaskRequest.getIdTask();
    }

}
