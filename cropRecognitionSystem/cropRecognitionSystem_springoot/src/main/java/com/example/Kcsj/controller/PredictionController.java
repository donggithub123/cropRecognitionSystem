package com.example.Kcsj.controller;

import com.alibaba.fastjson.JSONObject;
import com.example.Kcsj.common.Result;
import com.example.Kcsj.entity.Records;
import com.example.Kcsj.mapper.RecordsMapper;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import javax.annotation.Resource;
import java.util.Date;

@RestController
@RequestMapping("/flask")
public class PredictionController {
    @Resource
    RecordsMapper recordsMapper;

    private final RestTemplate restTemplate = new RestTemplate();

    // 定义接收的参数类
    public static class PredictRequest {
        private String model_name;
        private String weights;
        private String username;
        private String image_path;

        // Getters and setters
        public String getModel_name() {
            return model_name;
        }

        public void setModel_name(String model_name) {
            this.model_name = model_name;
        }

        public String getWeights() {
            return weights;
        }

        public void setWeights(String weights) {
            this.weights = weights;
        }

        public String getImage_path() {
            return image_path;
        }

        public void setImage_path(String image_path) {
            this.image_path = image_path;
        }

        public String getUsername() {
            return username;
        }

        public void setUsername(String username) {
            this.username = username;
        }
    }

    @PostMapping("/predict")
    public Result<?> predict(@RequestBody PredictRequest request) {
        if (request == null || request.getImage_path() == null || request.getImage_path().isEmpty()) {
            return Result.error("-1", "未提供图片链接");
        } else if (request.getWeights() == null || request.getWeights().isEmpty()) {
            return Result.error("-1", "未提供权重");
        } else if (request.getModel_name() == null || request.getModel_name().isEmpty()) {
            return Result.error("-1", "未提供模型");
        }

        // 输出模型名称和图片路径
        System.out.println("Received model name: " + request.getModel_name());
        System.out.println("Received weights: " + request.getWeights());
        System.out.println("Received image URL: " + request.getImage_path());

        try {
            // 创建请求体
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<PredictRequest> requestEntity = new HttpEntity<>(request, headers);

            // 调用 Flask API
            String response = restTemplate.postForObject("http://localhost:5000/predict", requestEntity, String.class);
            System.out.println("Received response: " + response);
            JSONObject responses = JSONObject.parseObject(response);
            Records records = new Records();
            records.setWeights(request.getWeights()); // 设置权重
            records.setModelName(request.getModel_name()); // 设置模型名称
            records.setImg(request.getImage_path()); // 设置图片路径
            records.setUsername(request.getUsername());
            records.setResult(String.valueOf(responses.get("prediction"))); // 设置预测结果
            records.setConfidence(String.valueOf(responses.get("confidence"))); // 设置置信度
            records.setTotalTime(String.valueOf(responses.get("total_time"))); // 设置置信度
            records.setTime(new Date()); // 设置创建时间
            recordsMapper.insert(records); // 插入到数据库
            return Result.success(response);
        } catch (Exception e) {
            return Result.error("-1", "Error: " + e.getMessage());
        }
    }

    @GetMapping("/file_names")
    public Result<?> getFileNames() {
        try {
            // 调用 Flask API
            String response = restTemplate.getForObject("http://127.0.0.1:5000/file_names", String.class);
            return Result.success(response);
        } catch (Exception e) {
            return Result.error("-1", "Error: " + e.getMessage());
        }
    }
}
