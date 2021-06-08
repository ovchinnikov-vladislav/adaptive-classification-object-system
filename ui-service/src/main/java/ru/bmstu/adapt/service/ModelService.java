package ru.bmstu.adapt.service;

import org.springframework.http.HttpStatus;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Service;
import org.springframework.ui.Model;

import java.time.LocalDate;

@Service
public class ModelService {

    public void prepareErrorModelForTemplate(Model model, Integer status, ServerHttpResponse response) {
        model.addAttribute("status", "Неопределенная ошибка");

        if (HttpStatus.valueOf(status) == HttpStatus.NOT_FOUND) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Страница или ресурс не найдены");
            response.setStatusCode(HttpStatus.NOT_FOUND);
        } else if (HttpStatus.valueOf(status) == HttpStatus.BAD_REQUEST) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Запрос недействителен");
            response.setStatusCode(HttpStatus.BAD_REQUEST);
        } else if (HttpStatus.valueOf(status).is5xxServerError()) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Фатальная ошибка (ошибка сервера)");
            response.setStatusCode(HttpStatus.valueOf(status));
        } else {
            response.setStatusCode(HttpStatus.INTERNAL_SERVER_ERROR);
        }
        model.addAttribute("additional_text",  "Похоже вы нашли баг в матрице...");
    }
}
