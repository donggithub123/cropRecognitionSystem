package com.example.Kcsj.controller;

import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.example.Kcsj.common.Result;
import com.example.Kcsj.entity.Records;
import com.example.Kcsj.mapper.RecordsMapper;
import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;

@RestController
@RequestMapping("/records")
public class RecordsController {
    @Resource
    RecordsMapper recordsMapper;

    @GetMapping("/all")
    public Result<?> GetAll() {
        return Result.success(recordsMapper.selectList(null));
    }

    @GetMapping
    public Result<?> findPage(@RequestParam(defaultValue = "1") Integer pageNum,
                              @RequestParam(defaultValue = "10") Integer pageSize,
                              @RequestParam(defaultValue = "") String search,
                              @RequestParam(defaultValue = "") String search1,
                              @RequestParam(defaultValue = "") String search2) {
        LambdaQueryWrapper<Records> wrapper = Wrappers.<Records>lambdaQuery();
        wrapper.orderByDesc(Records::getTime);
        if (StrUtil.isNotBlank(search)) {
            wrapper.like(Records::getUsername, search);
        }
        if (StrUtil.isNotBlank(search1)) {
            wrapper.like(Records::getTime, search1);
        }
        if (StrUtil.isNotBlank(search2)) {
            wrapper.like(Records::getResult, search2);
        }
        Page<Records> Page = recordsMapper.selectPage(new Page<>(pageNum, pageSize), wrapper);
        return Result.success(Page);
    }

    @DeleteMapping("/{id}")
    public Result<?> delete(@PathVariable int id) {
        recordsMapper.deleteById(id);
        return Result.success();
    }

    @PostMapping("/update")
    public Result<?> updates(@RequestBody Records records) {
        recordsMapper.updateById(records);
        return Result.success();
    }


    @PostMapping
    public Result<?> save(@RequestBody Records records) {
        recordsMapper.insert(records);
        return Result.success();
    }
}
